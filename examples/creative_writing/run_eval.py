import json
import os
from datetime import datetime
from agents import HotpotQADataset, Solution, SolutionConfig
from agents.evaluation import Case
from agents.optimization.utils import OptimUtils
from agents import CreativeWritingDataset

os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""
time_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = f"eval/{time_path}_creative_writing"


def get_score(dir_list):
    save_result_path = "eval/final_result.json"

    scores = []
    count = 0
    for dir in dir_list:
        print(f"\n\nthe dir is: {dir}")
        files_and_folders = os.listdir(dir)
        files = [f for f in files_and_folders if os.path.isfile(
            os.path.join(dir, f))]
        for file_name in files:
            if file_name == "result.json":
                continue
            with open(f"{dir}/{file_name}", "r", encoding='utf-8') as f:
                result = json.load(f)
                count += 1
                print(f"file_{count} path is: {dir}/{file_name}")
                scores.append(result["dataset_eval"]["score"])

    save_info = {
        "average_score": sum(scores) / len(scores),
        "scores": scores,
    }
    print("\n\nfinal result is: ")
    print(save_info)
    with open(save_result_path, "w", encoding='utf-8') as f:
        json.dump(save_info, f, ensure_ascii=False, indent=4)


def run_eval():

    dataset = CreativeWritingDataset(split="all")
    solution_config_path = ""

    solution = Solution(SolutionConfig(solution_config_path))
    case_list = []
    for i in range(len(dataset)):
        case_list.append(Case(dataset.get_case_dict(i)))

    OptimUtils.parallel_case_forward(
        case_list, solution, 8, save_path, dataset.evaluate)

    scores = []
    for case in case_list:
        scores.append(case.dataset_eval.standard_eval_result["average_score"])

    result = {
        "average_score": sum(scores) / len(scores),
    }
    print(result)

    with open(f"{save_path}/result.json", "w", encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # run eval for all cases
    run_eval()
