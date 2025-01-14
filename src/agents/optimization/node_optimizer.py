import copy
import logging
import json
import os.path
from functools import partial
from pathlib import Path

from agents import SOP, Solution, AgentTeamConfig, AgentTeam, SolutionConfig
from agents.agents.llm import LLMConfig, OpenAILLM
from agents.optimization.optimizer import Optimizer, OptimizerConfig
from agents.optimization.utils import OptimUtils
from agents.task.node import Node
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.evaluation import Case
from agents.optimization import prompt_formatter


class NodeOptimizerConfig(OptimizerConfig):
    """
    The NodeOptimizerConfig class is used to configure the node optimizer settings.
    It extends the OptimizerConfig class.
    """

    def __init__(self, config_path):
        """
        Initializes the NodeOptimizerConfig object.

        Args:
            config_path (str): The path to the configuration file.
        """
        super().__init__(config_path)

        # The parent class parses the common fields, here we only parse the fields for node_optimizer
        self.llm_config: dict = self.node_optimizer_setting_dict.get("llm_config")
        self.meta_prompt: dict = self.node_optimizer_setting_dict.get("meta_prompt")


class NodeOptimizer(Optimizer):
    """
    The NodeOptimizer class is used to optimize nodes based on the configuration.
    """

    def __init__(self, config: NodeOptimizerConfig, logger_name: str):
        """
        Initializes the NodeOptimizer object.

        Args:
            config (NodeOptimizerConfig): The node optimizer configuration.
            logger_name (str): The name of the logger.
        """
        super().__init__(config)
        self.config = config

        # llm
        self.llm_eval = (OpenAILLM(LLMConfig(config.llm_config)) if config.llm_config else None)

        # prompt
        self.both_prompt = config.meta_prompt.get("both")
        self.meta_backward = config.meta_prompt.get("backward")
        self.meta_optim = config.meta_prompt.get("optim")

        # logger
        self.logger = logging.getLogger(logger_name) if logger_name else logging.getLogger(__name__)

    def optimize(
            self,
            case_list: list[Case],
            solution: Solution,
            save_dir: Path,
            parallel_max_num,
    ):
        """
        Optimizes nodes in the given list of cases.

        Args:
            case_list (list[Case]): The list of cases to be optimized.
            solution (Solution): The solution to be optimized.
            save_dir (Path): The directory to save the results.
            parallel_max_num (int): The maximum number of parallel processes.

        Returns:
            tuple: The updated solution and optimization status.
        """
        self.logger.info("Start Node Optimization")
        saved_ori_solution = copy.deepcopy(solution)

        # backward
        backward_save_dir = save_dir / "case_after_backward"
        partial_funcs = [partial(self.backward, case, solution, backward_save_dir) for case in case_list]
        OptimUtils.parallel_execution(partial_funcs, max_workers=parallel_max_num)

        # optimization (based on the suggestions)
        op_info = self.optimize_node(case_list, solution)

        # Determine the optimization status based on the success of any node optimization
        op_status = any(info["optim_status"] for info in op_info.values())

        # if optimized successfully, update the AgentTeam
        if op_status:
            all_node_roles_description = {}
            for node_name, node in solution.sop.nodes.items():
                all_node_roles_description[node_name] = node.node_roles_description
            agent_team_config = AgentTeamConfig.generate_config(
                solution.task.task_description, all_node_roles_description)
            solution.agent_team = AgentTeam(agent_team_config)

        # save new solution and op_info
        try:
            with open(save_dir / "node_optim_info.json", "w", encoding="utf-8") as f:
                json.dump(op_info, f, ensure_ascii=False, indent=4)
            solution.dump(save_dir)
            solution = Solution(config=SolutionConfig(str(save_dir / "solution.json")))
        except Exception as e:
            self.logger.error(f"Error in saving solution: {e}")
            solution = saved_ori_solution
            solution.dump(save_dir / "accepted_solution")

        return solution, op_status

    def backward(self, case: Case, solution: Solution, save_dir: str):
        """
        Performs backward calculation for each node in the given case.

        Args:
            case (Case): The case for which the backward calculation is performed.
            solution (Solution): The solution associated with the case.
            save_dir (str): The directory to save the results.

        Returns:
            None
        """
        self.logger.info(f"Start backward for case: {case.case_id}")
        last_requirement_for_previous = case.loss.requirement_for_previous

        node_name_list = []
        for state in case.trajectory.states:
            if state.node.node_name not in node_name_list:
                node_name_list.append(state.node.node_name)

        state_idx = 0
        for op_node_name in node_name_list[::-1]:
            # get state idx of the op_node_name, it is the last state idx of the node
            for idx in range(len(case.trajectory.states) - 1, -1, -1):
                if case.trajectory.states[idx].node.node_name == op_node_name:
                    state_idx = idx
                    break

            # Call LLM to calculate backward for each node
            prompt = prompt_formatter.formulate_prompt_for_node_backward(
                self.meta_backward, case, solution.sop.nodes[op_node_name], last_requirement_for_previous)
            _, content = self.llm_eval.get_response(
                chat_messages=None, system_prompt="", last_prompt=prompt, stream=False)

            # Extract data from LLM response
            need_extract_key = self.meta_backward.get("extract_key", copy.deepcopy(
                ["analyse", "suggestion", "requirement_for_previous"]))
            backward_info_dict = OptimUtils.extract_data_from_response(
                content, need_extract_key, self.logger,
                "There are neither suggestions nor requirements.",
            )
            backward_info_dict["response"] = content
            backward_info_dict["prompt"] = prompt
            assert "suggestion" in backward_info_dict.keys(), "suggestion must be in the extracted data"
            assert "requirement_for_previous" in backward_info_dict.keys(), "requirement_for_previous must be in the extracted data"

            # update the state backward info, save the backward info to the case
            case.trajectory.states[state_idx].node_backward.update(**backward_info_dict)
            last_requirement_for_previous = backward_info_dict["requirement_for_previous"]

        # save the backward info to the file
        if save_dir:
            case.dump(os.path.join(save_dir, f"{case.case_id}.json"))

    def optimize_node(self, case_list: list[Case], solution: Solution):
        """
        Optimizes the configuration of each node in the solution based on the cases.

        Args:
            case_list (list[Case]): The list of cases.
            solution (Solution): The solution to be optimized.

        Returns:
            dict: A dictionary with the optimization status and method for each node.
        """
        node_name_list = []
        for state in case_list[0].trajectory.states:
            # Get unique node names based on case0's information and node_name is not duplicated
            if state.node.node_name not in node_name_list:
                node_name_list.append(state.node.node_name)

        # Do optimization for each node and get the optimization method
        op_info = {}
        for node_name in node_name_list:
            new_node, op_status, op_method = self.optimize_single_node(case_list, solution.sop.nodes[node_name])
            solution.sop.nodes[node_name] = new_node
            op_info[node_name] = {"optim_status": op_status, "optim_method": op_method}

        return op_info

    def optimize_single_node(self, case_list: list[Case], node: Node):
        """
        Optimizes the configuration information of a single node.

        Args:
            case_list (list[Case]): The list of cases.
            node (Node): The node to be optimized.

        Returns:
            tuple: The new Node object, a boolean indicating if the update was successful,
                   and the optimization method (JSON if successful, otherwise a string).
        """
        saved_node = copy.deepcopy(node)

        prompt = prompt_formatter.formulate_prompt_for_node_optim(self.meta_optim, node, case_list)
        _, content = self.llm_eval.get_response(chat_messages=None, system_prompt="", last_prompt=prompt, stream=False)

        # Modify the SOP configuration based on LLM suggestions
        extracted_dict = OptimUtils.extract_data_from_response(content, self.meta_optim["extract_key"])
        optim_method_str = extracted_dict.get("result")

        if optim_method_str == "" or optim_method_str == "[]":
            # current node performs well, no need to optimize
            self.logger.debug(f"No need to optimize node: {node.node_name}")
            return node, True, optim_method_str
        elif optim_method_str is None:
            self.logger.error(f"Error in optimizing node: {node.node_name}, the result is None.")
            return node, False, None
        else:
            self.logger.debug(f"try to optimize node: {node.node_name}")
            try:
                return self.do_node_optim(node, optim_method_str, self.logger)
            except Exception as e:
                self.logger.error(f"Error in do_node_optim: {e}")
                self.logger.error(f"optim_method_str: {optim_method_str}")
                # return the original node
                return saved_node, False, optim_method_str

    @staticmethod
    def do_node_optim(node: Node, optim_method_str: str, logger):
        """
        Executes the node optimization based on the given method.

        Note: If an error occurs during optimization, an exception is raised instead of returning,
        because the upper layer will handle the error and return the original unoptimized node.

        Args:
            node (Node): The node to be optimized.
            optim_method_str (str): The optimization method in JSON string format.
            logger: The logger for logging information and errors.

        Returns:
            tuple: The optimized Node object, a boolean indicating if the update was successful,
                   and the optimization method (JSON if successful, otherwise a string).
        """
        optim_method = json.loads(optim_method_str)

        # Validity check, attempt update only if successful
        check_status, reasons = NodeOptimizer.validate_dict(optim_method)

        if not check_status:
            logger.error(f"Error in validating optim_method: {reasons}, optim_method: {optim_method}")
            raise ValueError(f"Error in validating optim_method: {reasons}")
        else:
            logger.debug(f"succeed in validating optim_method, load the json successfully.")
            # try to optim the node
            for rule in optim_method:
                action = rule.get("action")
                if action == "add_role":
                    role_name = rule["role_name"]
                    role_description = rule["role_description"]
                    role_prompt = rule["role_prompt"]
                    role_prompt_key = "step_" + role_name
                    assert role_name not in node.node_prompt_paddings.keys(), f"Role name '{role_name}' already exists."
                    node.node_prompt_templates[role_prompt_key] = role_prompt
                    node.node_roles_description[role_name] = role_description
                    node.node_prompt_paddings[role_name] = {
                        role_prompt_key: {"value_source": "case", "value": "input_data"}}

                elif action == "delete_role":
                    role_name = rule["role_name"]
                    node.node_roles.pop(role_name)
                    node.node_roles_description.pop(role_name)
                    node.node_primary_prompts.pop(role_name)
                    node.node_prompt_templates.pop(role_name)
                    node.node_prompt_paddings.pop(role_name)
                    if len(node.node_prompt_paddings) == 0:
                        raise ValueError("The node should have at least one role.")
                    # begin role may need to be updated
                    if node.begin_role == role_name:
                        node.begin_role = next(iter(node.node_prompt_templates))

                elif action == "update_role_description":
                    role_name = rule["role_name"]
                    role_description = rule["role_description"]
                    node.node_roles_description[role_name] = role_description

                elif action == "update_controller":
                    node.controller.route_type = rule["route_type"]
                    node.controller.route_system_prompt = rule.get("route_system_prompt")
                    node.controller.route_last_prompt = rule.get("route_last_prompt")
                    assert ((node.controller.route_type in ["order", "random"]) or
                            (node.controller.route_type == "llm" and
                             node.controller.route_system_prompt and
                             node.controller.route_last_prompt)), \
                        "when route_type is llm, route_system_prompt and route_last_prompt should not be empty."

                elif action == "update_node_description":
                    node.node_description = rule["node_description"]

                else:
                    logger.error(f"Unknown action '{action}'")
                    raise ValueError(f"Unknown action, the optim rule is '{rule}'")

            # update the node successfully, all the rules are valid
            return node, True, optim_method

    @staticmethod
    def validate_dict(optim_method_list):
        """
        Validates the optimization method list.

        Args:
            optim_method_list (list): The list of optimization methods.

        Returns:
            tuple: A boolean indicating if the validation passed, and a string with the validation message.
        """
        for rule in optim_method_list:
            action = rule.get("action")
            if action == "add_role":
                required_keys = ["role_name", "role_description", "role_prompt"]
                for key in required_keys:
                    if key not in rule:
                        return False, f"Missing key '{key}' for action '{action}'"
                    if not isinstance(rule[key], str) or not rule[key]:
                        return False, f"Invalid value for key '{key}' for action '{action}'"
            elif action == "delete_role":
                required_keys = ["role_name"]
                for key in required_keys:
                    if key not in rule:
                        return False, f"Missing key '{key}' for action '{action}'"
                    if not isinstance(rule[key], str) or not rule[key]:
                        return False, f"Invalid value for key '{key}' for action '{action}'",

            elif action == "update_role_description":
                required_keys = ["role_name", "role_description"]
                for key in required_keys:
                    if key not in rule:
                        return False, f"Missing key '{key}' for action '{action}'"
                    if not isinstance(rule[key], str) or not rule[key]:
                        return False, f"Invalid value for key '{key}' for action '{action}'",

            elif action == "update_controller":
                required_keys = ["route_type", "route_system_prompt", "route_last_prompt"]
                for key in required_keys:
                    if key not in rule:
                        return False, f"Missing key '{key}' for action '{action}'"
                    if not isinstance(rule[key], str):
                        return False, f"Invalid value for key '{key}', the key value is {str(rule[key])}, action is '{action}'"
                    if key == "route_type" and rule[key] not in ["random", "order", "llm"]:
                        return False, f"Invalid value for key '{key}', the key value is {str(rule[key])}, action is '{action}'",
                    if key == "route_type" and rule[key] in ["random", "order"]:
                        # if route_type is random or order, "route_system_prompt", "route_last_prompt" are not needed
                        break

            elif action == "update_node_description":
                required_keys = ["node_description"]
                for key in required_keys:
                    if key not in rule:
                        return False, f"Missing key '{key}' for action '{action}'"
                    if not isinstance(rule[key], str) or not rule[key]:
                        return False, f"Invalid value for key '{key}' for action '{action}'",

            else:
                return False, f"Unknown action '{action}'"
        return True, "Validation passed"
