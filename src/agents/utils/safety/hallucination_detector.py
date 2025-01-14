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
import openai
import time

class HallucinationDetector:
    def __init__(self, api_key: str, model: str = "gpt-4", max_retries: int = 3):
        """
        Initialize a hallucination detector using OpenAI's ChatGPT API.

        Args:
            api_key (str): OpenAI API key for authentication.
            model (str): ChatGPT model to use (default: gpt-4).
            max_retries (int): Maximum number of retries for regeneration.
        """
        openai.api_key = api_key
        self.model = model
        self.max_retries = max_retries

    def detect_hallucination(self, prompt: str, response: str) -> bool:
        """
        Detect if the response contains hallucinations.

        Args:
            prompt (str): The original input prompt.
            response (str): The AI-generated response to analyze.

        Returns:
            bool: True if hallucination is detected, False otherwise.
        """
        verification_prompt = (
            "Please verify the factual accuracy of the following AI response based on the prompt. "
            "Indicate if there are hallucinations (made-up or false statements) in the response.\n\n"
            f"Prompt: {prompt}\n\nResponse: {response}\n\n"
            "Your answer should be 'Yes' if there are hallucinations, otherwise 'No'."
        )

        try:
            verification_response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "system", "content": "You are a fact-checking assistant."},
                         {"role": "user", "content": verification_prompt}],
                temperature=0,
            )
            verdict = verification_response['choices'][0]['message']['content'].strip().lower()
            return verdict == "yes"
        except Exception as e:
            print(f"Error during hallucination detection: {e}")
            return False

    def regenerate_until_valid(self, prompt: str, validation_prompt: str = None) -> str:
        """
        Regenerate responses until no hallucinations are detected.

        Args:
            prompt (str): The input prompt for the AI.
            validation_prompt (str): An optional custom prompt for validation.

        Returns:
            str: A valid AI response free of hallucinations.
        """
        for attempt in range(self.max_retries):
            try:
                # Generate a response
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )['choices'][0]['message']['content'].strip()

                print(f"Attempt {attempt + 1}: Response: {response}")

                # Detect hallucinations
                if validation_prompt:
                    detected = self.detect_hallucination(validation_prompt, response)
                else:
                    detected = self.detect_hallucination(prompt, response)

                if not detected:
                    print("No hallucinations detected.")
                    return response

                print("Hallucinations detected. Regenerating...")

            except Exception as e:
                print(f"Error during response generation: {e}")

            time.sleep(1)  # Avoid spamming the API

        raise ValueError("Exceeded maximum retries without generating a valid response.")


