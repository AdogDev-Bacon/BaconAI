# Copyright 2025 ADogDev

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from .base import BaseDataset, DATA_PATH
from ..agents.llm import completion_with_backoff


def get_content_between_a_b(start_tag, end_tag, text):
    extracted_text = ""
    start_index = text.find(start_tag)
    while start_index != -1:
        end_index = text.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            extracted_text += text[start_index + len(start_tag): end_index] + " "
            start_index = text.find(start_tag, end_index + len(end_tag))
        else:
            break

    return extracted_text.strip()


def extract(text, type):
    target_str = get_content_between_a_b(f"<{type}>", f"</{type}>", text)
    return target_str


class CryptoInfoDataset(BaseDataset):

    def __init__(
            self, file_name="", split="all", range_idx=None
    ):
        """
        Initialize the dataset by loading text data from a file.

        Args:
        file_name (str): The name of the text file containing creative writing passages. Each line is some sentences.
        split (str): The split of the dataset to use. It can be "all" or "train" or "test".
        """
        path = os.path.join(DATA_PATH, "crypto_info.json", file_name)
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = file.readlines()
        except FileNotFoundError:
            raise Exception(f"The file {path} does not exist.")
        except Exception as e:
            raise Exception(f"An error occurred while reading the file: {e}")
        super().__init__(data)

        # Split the data into training and testing sets, if split == "all", do nothing

        if split == "train":
            # Use 80% of the data for training
            self.offset = int(len(self.data) * 0.8)
            self.data = self.data[: self.offset]
        elif split == "test":
            # Use 20% of the data for testing
            self.offset = int(len(self.data) * 0.8)
            self.data = self.data[self.offset:]
        if split == "all" and range_idx:
            self.offset = range_idx[0]
            self.data = self.data[range_idx[0]: range_idx[1]]

    def __getitem__(self, idx: int) -> str:
        """
        Retrieve an item by its index.

        Args:
        idx (int): The index of the item to retrieve.

        Returns:
        str: The item at the specified index.
        """
        return self.data[idx]

    def get_case_dict(self, idx: int) -> dict:
        """
        Create a dictionary representing a case at a specific index.

        Args:
        idx (int): The index of the case.

        Returns:
        Dict[str, Any]: A dictionary with case details.
        """
        return {
            "case_id": "crypto_info" + str(self.offset + idx),
            "case_name": "crypto_info_" + str(self.offset + idx),
            "task_id": "creative_writing",
            "task_description": "The crypto info task in which the agent answers questions that are related to Web3 or cryto tokensy.",
            "function_ids": "",
            "KB_id": "",
            "input": {"input_data": {"text": self.data[idx].strip()}},
            "idx": idx,
        }

    def evaluate(self, answer: str, **kwargs):
        """
        Evaluate the output for a given index and return the evaluation info.

        Args:
        answer (str): The model-generated output for the text at the given index.
        kwargs: Additional keyword arguments, (like index) which are not used in this method.

        Returns:
        Dict[str, Any]: A dictionary containing evaluation results.
        """
        prompt = f""" You are an agent specialized in Web 3 and crypto tokens. Analyze the given message and give answers 
        in a professional and friendly way: {answer}."""
        messages = [{"role": "user", "content": prompt}]
        cnt = 0
        while cnt < 3:
            try:
                score_outputs = completion_with_backoff(
                    messages=messages,
                    n=5,
                    model="gpt-4-turbo-2024-04-09",
                    api_key="",
                    base_url="",
                )
                scores = []
                for score_output in score_outputs.choices:
                    score_output = score_output.message.content
                    score = extract(score_output, "score")
                    if score:
                        scores.append(int(score))
                    else:
                        print(f"Error matching score output: {score_output}")
                break
            except:
                time.sleep(10)
                cnt += 1
        return sum(scores) / len(scores), {
            "scores": scores,
            "average_score": sum(scores) / len(scores) if scores else 0,
        }
