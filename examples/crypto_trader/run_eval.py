import json
import os
from datetime import datetime
from agents import CryptoTradeDataset, Solution, SolutionConfig
from agents.evaluation import Case
from agents.optimization.utils import OptimUtils

os.environ["API_KEY"] = ""
os.environ["BASE_URL"] = ""
time_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = f"eval/{time_path}_crypto_trader"

def get_score(dir_list):
    save_result_path = "eval/final_crypto_results.json"

    scores = []
    count = 0
    for dir in dir_list:
        print(f"\n\nProcessing directory: {dir}")
        files_and_folders = os.listdir(dir)
        files = [f for f in files_and_folders if os.path.isfile(os.path.join(dir, f))]
        for file_name in files:
            if file_name == "result.json":
                continue
            with open(f"{dir}/{file_name}", "r", encoding='utf-8') as f:
                result = json.load(f)
                count += 1
                print(f"File {count} path: {dir}/{file_name}")
                scores.append(result["average_profit"])

    save_info = {
        "average_profit": sum(scores) / len(scores) if scores else 0,
        "profits": scores,
    }
    print("\n\nFinal evaluation result:")
    print(save_info)
    with open(save_result_path, "w", encoding='utf-8') as f:
        json.dump(save_info, f, ensure_ascii=False, indent=4)

def run_eval():

    dataset = CryptoTradeDataset()
    solution_config_path = ""  # Path to the solution configuration

    solution = Solution(SolutionConfig(solution_config_path))
    case_list = []
    for i in range(len(dataset)):
        case_list.append(Case(dataset.get_case_dict(i)))

    OptimUtils.parallel_case_forward(
        case_list, solution, 8, save_path, dataset.evaluate
    )

    profits = []
    for case in case_list:
        profits.append(case.dataset_eval.standard_eval_result["actual_profit"])

    result = {
        "average_profit": sum(profits) / len(profits) if profits else 0,
    }
    print("\n\nEvaluation results:")
    print(result)

    with open(f"{save_path}/result.json", "w", encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Run evaluation for all cases
    run_eval()
