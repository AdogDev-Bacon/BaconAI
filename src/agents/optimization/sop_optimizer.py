import copy
import logging
import json
from pathlib import Path

from agents import SOP, Solution, SolutionConfig
from agents.agents.llm import LLMConfig, OpenAILLM
from agents.evaluation import Case
from agents.task.node import NodeConfig, Node
from agents.optimization import prompt_formatter
from agents.optimization.optimizer import Optimizer, OptimizerConfig
from agents.optimization.utils import OptimUtils

logger = logging.getLogger(__name__)


class SOPOptimizerConfig(OptimizerConfig):
    """
    Configuration for the SOP Optimizer.

    This class parses and stores the configuration settings specific to the SOP optimizer,
    extending the generic OptimizerConfig.

    Args:
        config_path (str): Path to the configuration file.
    """

    def __init__(self, config_path):
        super().__init__(config_path)

        self.llm_config = self.sop_optimizer_setting_dict.get("llm_config")
        self.meta_prompt = self.sop_optimizer_setting_dict.get("meta_prompt")


class SOPOptimizer(Optimizer):
    """
    The SOP Optimizer class for optimizing SOPs.

    This class extends the generic Optimizer and uses the SOPOptimizerConfig to set up
    SOP optimization, including handling large language model (LLM) configurations and
    managing optimization components.

    Args:
        config (SOPOptimizerConfig): Configuration instance for the SOP optimizer.
        logger_name (str): Name of the logger.
    """

    def __init__(self, config: SOPOptimizerConfig, logger_name: str):
        super().__init__(config)
        self.config = config

        # llm
        self.llm_eval = OpenAILLM(LLMConfig(config.llm_config)) if config.llm_config else None

        # prompt
        self.meta_backward = config.meta_prompt.get("backward")
        self.meta_optim = config.meta_prompt.get("optim")

        # logger
        self.logger = logging.getLogger(logger_name) if logger_name else logging.getLogger(__name__)

    def optimize(
            self,
            case_list: list[Case],
            solution: Solution,
            save_path: Path,
            parallel_max_num):
        """
        Optimizes the SOP for the given list of cases.

        Args:
            case_list (list[Case]): List of cases to be optimized.
            solution (Solution): The solution to be optimized.
            save_path (Path): Path to save the results.
            parallel_max_num (int): Maximum number of parallel processes.

        Returns:
            tuple: The updated solution and optimization status.
        """
        self.logger.info("Start to optimize SOP")

        # 1. Perform backward pass for each case to get the necessary information
        for case in case_list:
            OptimUtils.node_eval(case, solution, self.llm_eval, self.logger)
            self.backward(case, solution, save_path / "backward")

        # 2. Construct the prompt and get the optimization method
        prompt = prompt_formatter.formulate_prompt_for_sop_optim(self.meta_optim, solution.sop, case_list)
        _, content = self.llm_eval.get_response(chat_messages=None, system_prompt="",
                                                last_prompt=prompt, stream=False)

        # print("in optimize sop, prompt is: ", prompt)
        # print("in optimize sop, response is : ", content)

        # 3. Extract results and attempt to optimize the SOP
        extracted_dict = OptimUtils.extract_data_from_response(content, self.meta_optim["extract_key"])
        result = extracted_dict["result"]
        analyse = extracted_dict["analyse"]
        solution, op_status = SOPOptimizer.try_optim_with_llm_result(solution, result, self.logger)

        # Default optimized solution: controller's transit_type is llm, transit_system_prompt and transit_last_prompt are empty
        if op_status:
            for node in solution.sop.nodes.values():
                node.controller.update({"transit_type": "llm", "transit_system_prompt": "", "transit_last_prompt": ""})

        # 4. Save the final solution and optimization information
        optim_info = {
            "optim_status": op_status,
            "result": result,
            "analyse": analyse,
            "prompt": prompt,
            "response": content,
        }
        with open(save_path / "sop_optim_info.json", "w", encoding="utf-8") as f:
            json.dump(optim_info, f, ensure_ascii=False, indent=4)

        # Reload the saved solution to avoid mismatches between config and actual data
        solution.dump(save_path)
        solution = Solution(config=SolutionConfig(f"{save_path}/solution.json"))
        self.logger.debug(
            f"Finish optimizing SOP, save the final solution to: {save_path}/solution.json. The optim_status is {op_status}"
        )
        return solution, op_status

    def backward(self, case: Case, solution: Solution, save_dir: Path = None):
        """
        Performs backward pass to evaluate the case and get suggestions for optimizing the SOP.

        Args:
            case (Case): The case to be evaluated.
            solution (Solution): The solution configuration.
            save_dir (Path, optional): The directory to save the results. Defaults to None.
        """
        prompt = prompt_formatter.formulate_prompt_for_sop_optim(
            self.meta_backward, solution.sop, [case], consider_case_loop=False)
        _, content = self.llm_eval.get_response(
            chat_messages=None, system_prompt="", last_prompt=prompt, stream=False)

        assert "suggestion" in content, f"Content does not contain suggestion field, content is {content}"
        extracted_dict = OptimUtils.extract_data_from_response(content, self.meta_backward["extract_key"])
        suggestion = extracted_dict["suggestion"]
        analyse = extracted_dict.get("analyse", "")
        suggestion_data = {
            "prompt": prompt,
            "response": content,
            "suggestion": suggestion,
            "analyse": analyse,
        }
        case.sop_suggestion.update(**suggestion_data)

        if save_dir:
            case.dump(save_dir / f"{case.case_id}.json")

    @staticmethod
    def try_optim_with_llm_result(solution: Solution, result: str, logger):
        """
        Attempts to optimize the solution using the given LLM result.

        Args:
            solution (Solution): The current solution.
            result (str): The optimization result provided by the large language model.
            logger (logging.Logger): Logger for recording information.

        Returns:
            tuple: The optimized solution and a boolean indicating the success of the optimization.
        """
        # result is empty, no optimization attempt
        if not result:
            logger.info("result is None or empty, can not optimize SOP")
            return solution, False

        # result is not empty, optimization can be attempted
        logger.debug("In SOP optim, LLM result is not None, try to use the result to optimize SOP")
        logger.debug(f"result is: {result}")

        # Attempt to parse the result and check the legality of the op_list
        op_status, reason, op_list = SOPOptimizer.check_sop_optim_op_list_legal(result, solution.sop, logger)

        # If the optimization method is not legal, no optimization attempt
        if not op_status:
            return solution, False

        # If the optimization method is legal, attempt optimization
        else:
            logger.debug("op_list is legal, try to optimize SOP")

            # Deepcopy the solution before optimization for rollback in case of errors
            deep_copy_solution = copy.deepcopy(solution)
            idx = -1
            op = None
            # Attempt each optimization, rollback if any error occurs
            try:
                for idx, op in enumerate(op_list):
                    SOPOptimizer.do_sop_optim(solution, op, logger)
            except Exception as e:
                logger.error(f"Error when optimizing SOP: {e}, the op idx is {idx}, the op is {op}")
                # On error, rollback and return the original solution with False status
                return deep_copy_solution, False
            # On successful optimization, return the optimized solution and True status
            return solution, True

    @staticmethod
    def do_sop_optim(solution: Solution, op: dict, logger):
        """
        Performs SOP optimization based on the given operation.

        Args:
            solution (Solution): The current solution.
            op (dict): The optimization operation.
            logger (logging.Logger): Logger for recording information.
        """
        # Perform optimization on the specific operation
        sop: SOP = solution.sop
        action = op["action"]
        if action == "add_node":
            new_node_name = op["node_name"]
            node_config = NodeConfig.generate_config(
                task_description=solution.task.task_description,
                node_name=new_node_name,
                node_description=op["node_description"],
                next_nodes=op["edges"][new_node_name],
            )
            # Create the new node
            new_node = Node(node_config)
            new_node.next_nodes = {}
            for next_node_name in op["edges"][new_node_name]:
                new_node.next_nodes[next_node_name] = sop.nodes[next_node_name]
            new_node.controller.update(op["controller"])

            # Add the new node to SOP, only updating the current node's information
            sop.nodes[new_node.node_name] = new_node
        elif action == "delete_node":
            op_node_name = op["node_name"]
            sop.nodes.pop(op_node_name)
            for pre_node_name, next_node_name_list in op["edges"].items():
                sop.nodes[pre_node_name].next_nodes = {}
                for next_node_name in next_node_name_list:
                    sop.nodes[pre_node_name].next_nodes[next_node_name] = sop.nodes[next_node_name]
        elif action == "update_node_description":
            op_node_name = op["node_name"]
            op_node_description = op["node_description"]
            sop.nodes[op_node_name].node_description = op_node_description
        elif action == "update_edges":
            updated_edges = op["edges"]
            for pre_node_name, next_node_name_list in updated_edges.items():
                sop.nodes[pre_node_name].next_nodes = {}
                for next_node_name in next_node_name_list:
                    sop.nodes[pre_node_name].next_nodes[next_node_name] = sop.nodes[next_node_name]
        else:
            raise ValueError(f"Unknown action {action}")

    @staticmethod
    def check_sop_optim_op_list_legal(optim_result: str, sop: SOP, logger):
        """
        Checks the legality of the SOP optimization operations provided by the LLM.

        Args:
            optim_result (str): The optimization result provided by the LLM.
            sop (SOP): The current SOP.
            logger (logging.Logger): Logger for recording information.

        Returns:
            tuple: A boolean indicating legality, a string message, and the op_list.
        """
        try:
            op_list = json.loads(optim_result)
        except Exception as e:
            logger.debug(f"Error when loading the result as JSON: {e}, the result is: {optim_result}")
            return False, "Unable to convert the optimization result string to JSON format", None

        new_node_names = ["end_node"]
        try:
            logger.debug(f"load op_list successfully, op_list is: {op_list}")
            for idx, op in enumerate(op_list):
                assert "action" in op, f"The action field is missing in operation {idx + 1}"
                action = op["action"]
                if action == "add_node":
                    assert "node_name" in op, f"The node_name field is missing in add_node operation {idx + 1}: {op}"
                    assert "node_description" in op, f"The node_description field is missing in add_node operation {idx + 1}: {op}"
                    assert "edges" in op, f"The edges field is missing in add_node operation {idx + 1}: {op}"
                    assert op["node_name"] in op[
                        "edges"], f"In add_node, the new node must have successor nodes defined, operation {idx + 1}: {op}"
                    new_node_names.append(op["node_name"])

                elif action == "delete_node":
                    assert "node_name" in op, f"The node_name field is missing in delete_node operation {idx + 1}: {op}"
                    assert "edges" in op, f"The edges field is missing in delete_node operation {idx + 1}: {op}"

                elif action == "update_node_description":
                    assert "node_name" in op, f"The node_name field is missing in update_node_description operation {idx + 1}: {op}"
                    assert "node_description" in op, f"The node_description field is missing in update_node_description operation {idx + 1}: {op}"

                elif action == "update_edges":
                    updated_edges = op["edges"]
                    for node_name in updated_edges:
                        assert node_name in sop.nodes or node_name in new_node_names, f"The from_node_name in update_edges operation is not in SOP, the from node is {node_name}, operation {idx + 1}: {op}"
                        for next_node_name in updated_edges[node_name]:
                            assert next_node_name in sop.nodes or next_node_name in new_node_names, f"The next_node_name in update_edges operation is not in SOP, the next node is {next_node_name}, operation {idx + 1}: {op}"

        except AssertionError as ae:
            logger.error(f"Assertion error during SOP optimization result legality check: {ae}")
            logger.error(
                f"New node names: {new_node_names}, SOP node names: {[node.node_name for node in sop.nodes.values()]}")
            return False, str(ae), op_list
        except Exception as e:
            logger.error(f"Unexpected error during SOP optimization result legality check: {e}")
            return False, f"Unexpected assertion error: {e}", op_list
        return True, "legal_op_list", op_list