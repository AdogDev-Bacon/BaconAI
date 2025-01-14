import copy
import json
import logging
import os.path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.optimization.optimizer import Optimizer
from agents.utils.prompts import DEFAULT_NODE_PROMPT_TEMPLATES
from agents.evaluation import Case
from agents.agents.llm import LLMConfig, OpenAILLM
from agents.task.solution import SolutionConfig, Solution
from agents.optimization.optimizer import OptimizerConfig
from agents.optimization.utils import OptimUtils
from agents.optimization import prompt_formatter


class PromptOptimizerConfig(OptimizerConfig):
    """
    Configuration for the Prompt Optimizer.

    This class parses and stores the configuration settings specific to the prompt optimizer,
    extending the generic OptimizerConfig.

    Args:
        config_path (str): Path to the configuration file.

    Attributes:
        allow_delete_template_variable (bool): Whether to allow deleting template variables. Defaults to False.
        llm_config (dict): Configuration dictionary for the large language model.
        meta_prompt (dict): Configuration dictionary for the meta prompt.
        needed_optim_component (list): List of components that need optimization.
        needed_optim_padding (bool): Whether padding is needed for optimization. Defaults to False.
    """

    def __init__(self, config_path):
        super().__init__(config_path)

        # The parent class parses the part of the fields that are public, and only the fields that prompt_optimizer are parsed here
        self.allow_delete_template_variable: bool = self.prompt_optimizer_setting_dict.get(
            "allow_delete_template_variable", False
        )
        self.llm_config: dict = self.prompt_optimizer_setting_dict.get("llm_config")
        self.meta_prompt: dict = self.prompt_optimizer_setting_dict.get("meta_prompt")
        self.needed_optim_component: list = self.prompt_optimizer_setting_dict.get("needed_optim_component")
        self.needed_optim_padding: bool = self.prompt_optimizer_setting_dict.get("needed_optim_padding", False)


class PromptOptimizer(Optimizer):
    """
    The Prompt Optimizer class for optimizing prompts.

    This class extends the generic Optimizer and uses the PromptOptimizerConfig to set up
    prompt optimization, including handling large language model (LLM) configurations and
    managing optimization components.

    Args:
        config (PromptOptimizerConfig): Configuration instance for the prompt optimizer.
        logger_name (str, optional): Name of the logger. Defaults to None.
    """

    def __init__(self, config: PromptOptimizerConfig, logger_name=None):
        super().__init__(config)
        self.config = config

        # Specifies whether to delete variables in the prompt template
        self.allow_delete_template_variable = config.allow_delete_template_variable
        assert (not self.has_ground_truth) or (
                self.has_ground_truth and self.has_result
        )

        # prompt
        self.meta_backward = config.meta_prompt["backward"]
        self.meta_optim = config.meta_prompt["optimization"]

        # optim_component
        self.needed_optim_component = config.needed_optim_component
        self.needed_optim_padding = config.needed_optim_padding

        # llm
        llm_config = LLMConfig(config.llm_config)
        self.llm_eval = OpenAILLM(llm_config) if llm_config else None

        # log
        self.logger = logging.getLogger(logger_name) if logger_name else logging.getLogger(__name__)

    def optimize(self, case_list: list[Case], solution: Solution, save_step_path, parallel_max_num=8):
        """
        Optimizes the prompts for a list of cases and updates the solution accordingly.

        Args:
            case_list (list[Case]): The list of cases to be optimized.
            solution (Solution): The solution to be optimized.
            save_step_path (str): The path to save the results of each step.
            parallel_max_num (int): The maximum number of parallel processes. Defaults to 8.

        Returns:
            tuple: The updated solution and optimization status.
        """
        step_info = {"score_before_optim": [case.loss.score for case in case_list]}
        step_info["average_score_before_optim"] = sum(step_info["score_before_optim"]) / len(case_list)

        # step2: get loss(the difference between result and ground_truth) and calculate the gradient of the loss
        self.logger.debug("In prompt optimizer, start backward")
        self.parallel_backward(case_list, parallel_max_num, save_step_path / "case_after_backward")
        self.logger.debug("In prompt optimizer, backward finished")

        # step3: update sop's prompt_template
        self.logger.debug("In prompt optimizer, start optimize prompt")
        optim_info_list = self.optimize_prompt(case_list)

        # attempt to optimize the solution based on optim_info_list
        optim_status = self.try_optim_prompt(case_list, solution, optim_info_list)

        # Determine the version of solution and case accepted by the current step
        for case in case_list:
            case.dump(os.path.join(str(save_step_path / "case_final"), f"{case.case_id}.json"))
        solution.dump(save_step_path)
        solution = Solution(config=SolutionConfig(f"{save_step_path}/solution.json"))

        # save the step info
        self.save_step(
            save_step_path,
            optim_info_list,
            step_info,
            optim_status,
        )
        return solution, optim_status

    def backward(self, case: Case, save_dir: str = None):
        """
        Performs backward optimization for the given case to get the gradient of the loss.

        Args:
            case (Case): The case to be optimized.
            save_dir (str, optional): The directory to save the results. Defaults to None.
        """
        # Transfer the result from loss, specifically the requirements/suggestions for the previous state, to the last state
        last_requirement_for_previous = case.loss.requirement_for_previous

        # Process each state to get modification suggestions, from back to front
        for s_idx, state in enumerate(case.trajectory.states[::-1]):
            # Get the state data and fill it into backward_prompt_temp
            state_index = len(case.trajectory.states) - s_idx - 1
            state_data = state.get_dict_for_trainer(
                ["prompt_template", "response", "prompt_components", "last_prompt_str", "prompts_order"])

            # Needed optimization components, if padding is needed, add all keys from prompt_template
            state_data["needed_optim_component"] = self.needed_optim_component[:]
            state_data["needed_optim_component"].extend(
                state_data["prompt_template"].keys()) if self.needed_optim_padding else None
            state_data["requirement_for_previous"] = last_requirement_for_previous
            state_data["previous_output"] = (
                case.trajectory.states[state_index - 1].action.content
                if state_index > 0
                else "This is the first node, no previous output."
            )
            state_data["content"] = state.action.content

            # Construct the prompt and call the model
            backward_prompt = prompt_formatter.formulate_prompt(self.meta_backward, state_data)

            # Call the large model to evaluate and extract results
            _, content = self.llm_eval.get_response(
                chat_messages=None,
                system_prompt="",
                last_prompt=backward_prompt,
                stream=False,
            )

            backward_info_dict = OptimUtils.extract_data_from_response(
                content,
                self.meta_backward["extract_key"],
                self.logger,
                "There are neither suggestions nor requirements.",
            )
            backward_info_dict["response"] = content
            backward_info_dict["prompt"] = backward_prompt
            assert "suggestion" in backward_info_dict.keys(), "suggestion must be in the extracted data"
            assert "requirement_for_previous" in backward_info_dict.keys(), "requirement_for_previous must be in the extracted data"

            # Update last_state_requirement, i.e., current state's requirements/suggestions for the previous state
            state.backward.update(**backward_info_dict)
            last_requirement_for_previous = state.backward.requirement_for_previous

        # save the case
        if save_dir:
            case.dump(os.path.join(save_dir, f"{case.case_id}.json"))

    def parallel_backward(
            self, case_list: list[Case], parallel_max_num, save_case_dir=None
    ):
        """
        Performs parallel backward optimization for a list of cases.

        Args:
            case_list (list[Case]): The list of cases to be optimized.
            parallel_max_num (int): The maximum number of parallel processes.
            save_case_dir (str, optional): The directory to save the cases after backward. Defaults to None.
        """
        with ThreadPoolExecutor(max_workers=parallel_max_num) as executor:
            futures = [
                executor.submit(self.backward, case, str(save_case_dir))
                for case in case_list
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error processing case: {e}")

    def optimize_prompt(self, case_list: list[Case]):
        """
        Optimizes the agent's prompt to minimize loss.

        Args:
            case_list (list[Case]): A batch of cases.

        Returns:
            list: A list of optimization information.
        """
        optim_info_list = []

        # Start optimizing from the last state, therefore it's a double loop: outer is state, inner is case
        case0_state_num = len(case_list[0].trajectory.states)
        for state_idx in range(case0_state_num - 1, -1, -1):
            # Call the large model for optimization and store the results in sop
            optim_prompt = prompt_formatter.formulate_prompt_for_prompt_optim(
                self.meta_optim, case_list, state_idx, self.needed_optim_component)
            _, content = self.llm_eval.get_response(
                chat_messages=None, system_prompt="", last_prompt=optim_prompt, stream=False)


            extract_dict = OptimUtils.extract_data_from_response(content, self.meta_optim["extract_key"])

            cur_optim_info = {
                "new_prompt": extract_dict["new_prompt"],
                "analyse": extract_dict.get("analyse"),
                "suggestion": [
                    case.trajectory.states[state_idx].backward.suggestion
                    for case in case_list
                ],
            }
            optim_info_list.insert(0, cur_optim_info)

        return optim_info_list

    def try_optim_prompt(self, case_list: list[Case], solution, optim_info_list):
        """
        Checks each prompt for validity and updates the solution if valid.

        Args:
            case_list (list[Case]): The list of cases.
            solution (Solution): The solution to be updated.
            optim_info_list (list): The list of optimization information.

        Returns:
            bool: The optimization status.
        """
        optim_status = False
        for state_index, cur_optim_info in enumerate(optim_info_list):
            # Save new template if valid
            new_template = cur_optim_info["new_prompt"]
            node_name = case_list[0].trajectory.states[state_index].node.node_name
            role_name = case_list[0].trajectory.states[state_index].action.agent_role

            # Get old configuration for comparison and validity check
            used_padding_template = case_list[0].trajectory.states[state_index].action.used_prompt_templates
            used_primary_template = solution.sop.nodes[node_name].node_primary_prompts[role_name]
            last_used_template = {**used_padding_template, **used_primary_template}

            check_result, new_template_dict, old_prompt_dict = (
                self.check_if_new_prompt_legal(
                    new_template,
                    last_used_template,
                    self.allow_delete_template_variable,
                    self.logger,
                )
            )

            if check_result:
                optim_status = True  # There is at least one valid optimization result
                self.update_prompt(solution, node_name, role_name, new_template_dict)

                self.logger.info(f"The new prompt for state_idx {state_index} is valid and updated in the solution.")
                self.logger.debug(f"The new prompt: {str(new_template_dict)}")
                self.logger.debug(f"The old prompt: {str(old_prompt_dict)}")
            else:
                self.logger.info(f"The new prompt for state_idx {state_index} is invalid.")
                self.logger.debug(f"The invalid new prompt: {str(new_template_dict)}")
        return optim_status

    def update_prompt(self, solution, node_name, role_name, new_template_dict):
        """
        Updates the prompt in the solution.

        Args:
            solution (Solution): The solution to be updated.
            node_name (str): The name of the node.
            role_name (str): The name of the role.
            new_template_dict (dict): The new template dictionary.
        """
        for key in new_template_dict:
            node = solution.sop.nodes[node_name]
            if key in ["TASK", "RULE", "STYLE", "EXAMPLE", "COT"]:
                node.node_primary_prompts[role_name][key] = new_template_dict[key]
            else:
                node.node_prompt_templates[key] = new_template_dict[key]

    @staticmethod
    def save_step(step_save_path, optim_info_list, step_info, optim_status):
        """
        Saves the final optimization results.

        Args:
            step_save_path (str): The path to save the step information.
            optim_info_list (list): The list of optimization information.
            step_info (dict): The step information.
            optim_status (bool): The optimization status.
        """
        step_info.update({"optim_status": optim_status, "optim_info": optim_info_list})

        with open(step_save_path / "step_info.json", "a", encoding="utf-8") as f:
            json.dump(step_info, f, ensure_ascii=False, indent=4)

    @staticmethod
    def check_if_new_prompt_legal(
            new_prompt_dict_str: str,
            old_prompt_dict: dict,
            allow_delete_template_variable: bool = False,
            logger=None,
    ):
        """
        Check if the new prompt is legal.

        This function checks whether the new prompt (given as a JSON string) is a valid update
        to the old prompt. It ensures that the new prompt's keys are a subset of the old prompt's keys
        and that variable counts are appropriate based on the `allow_delete_template_variable` flag.

        Args:
            new_prompt_dict_str (str): The new prompt as a JSON string, corresponding to the
                `node_prompt_templates` field in the SOP.
            old_prompt_dict (dict): The old prompt as a dictionary, corresponding to the
                `node_prompt_templates` field in the SOP.
            allow_delete_template_variable (bool, optional): Whether deleting variables in the template is allowed.
                Defaults to False.
            logger (Logger, optional): Logger instance for logging information and errors. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - bool: Indicates whether the new prompt is legal.
                - dict or None: The new prompt as a dictionary if it is legal, otherwise None.
                - dict: The old prompt dictionary.
        """
        old_prompt_dict = copy.deepcopy(old_prompt_dict)
        if not new_prompt_dict_str or new_prompt_dict_str == "{}":
            logger.info("The new prompt is empty or just an empty bracket '{}', not optimizing.")
            return False, {}, old_prompt_dict

        # Handle special character issues and convert the string to a JSON dictionary
        try:
            fixed_new_prompt_dict_str = OptimUtils.escape_special_chars_in_json_string(new_prompt_dict_str)
            if fixed_new_prompt_dict_str != new_prompt_dict_str:
                logger.info(
                    "Special characters found in the new prompt and have been escaped. The escaped prompt is: " + fixed_new_prompt_dict_str)
            new_prompt_dict = json.loads(fixed_new_prompt_dict_str)
        except json.JSONDecodeError:
            logger.error("The new prompt is not a valid JSON format: " + new_prompt_dict_str)
            return False, None, old_prompt_dict

        # Ensure the keys in the new prompt are a subset of the keys in the old prompt
        if not set(new_prompt_dict.keys()).issubset(set(old_prompt_dict.keys())):
            logger.error(
                "New prompt dictionary contains new keys. New prompt keys: " + str(
                    new_prompt_dict.keys()) + ", Old prompt keys: " + str(old_prompt_dict.keys()))
            return False, None, old_prompt_dict

        # Iterate and check if each prompt is valid
        for key in new_prompt_dict.keys():
            new_prompt = new_prompt_dict[key]
            old_prompt = old_prompt_dict[key]

            old_variables = prompt_formatter.get_config_needed_variables(
                {"old_prompt": old_prompt}, specific_key_list=["old_prompt"]
            )
            new_variables = prompt_formatter.get_config_needed_variables(
                {"new_prompt": new_prompt}, specific_key_list=["new_prompt"]
            )

            # 1. Check if there are additional variables
            if len(new_variables) > len(old_variables):
                logger.error(
                    "The number of variables in the new prompt exceeds that in the old prompt. New prompt variables: " +
                    str(new_variables) + ", Old prompt variables: " + str(old_variables))
                return False, None, old_prompt_dict

            # 2. Check if there are deleted variables
            if len(new_variables) < len(old_variables):
                if not allow_delete_template_variable:
                    if logger:
                        logger.error(
                            "The number of variables in the new prompt is less than that in the old prompt, and allow_delete_template_variable is False. New prompt variables: " + str(
                                new_variables) + ", Old prompt variables: " + str(old_variables))
                    return False, None, old_prompt_dict
                else:
                    if logger:
                        logger.info(
                            "The number of variables in the new prompt is less than that in the old prompt, but deleting template variables is allowed. New prompt variables: " + str(
                                new_variables) + ", Old prompt variables: " + str(old_variables))

            # 3. Check if there are modified variables
            for var in new_variables:
                if var not in old_variables:
                    if logger:
                        logger.error(
                            f"Variable {var} in the new prompt is not in the old prompt. New prompt: {new_prompt}, Old prompt: {old_prompt}")
                    return False, None, old_prompt_dict

        return True, new_prompt_dict, old_prompt_dict