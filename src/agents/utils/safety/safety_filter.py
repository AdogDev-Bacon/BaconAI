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
import torch
from transformers import LlamaForSequenceClassification, LlamaTokenizer

class ExternalSafetyFilter:
    def __init__(self, model_name, threshold: float = 0.5):
        """
        Initialize a safety filter using a LLaMA model for NSFW content detection.

        Args:
            model_name (str): Pretrained LLaMA model name for sequence classification.
            threshold (float): Threshold for classifying content as NSFW.
        """
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForSequenceClassification.from_pretrained(model_name)
        self.threshold = threshold

    def is_nsfw(self, text: str) -> bool:
        """
        Detect if the given text is NSFW.

        Args:
            text (str): Input text to classify.

        Returns:
            bool: True if the content is NSFW, False otherwise.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Run the model to get logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1).squeeze()
        nsfw_score = probs[1].item()

        return nsfw_score > self.threshold

