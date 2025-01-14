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
from .base import BaseRetriever
from transformers import RagRetriever, RagSequenceForGeneration, RagTokenizer

class RetrievalAugmentedGeneration(BaseRetriever):
    def __init__(self, retriever_index_path: str, model_name: str = "facebook/rag-sequence-nq"):
        """
        Initialize a Retrieval-Augmented Generation (RAG) pipeline.

        Args:
            retriever_index_path (str): Path to the retriever's index files.
            model_name (str): Pretrained RAG model name from Hugging Face.
        """
        self.tokenizer = RagTokenizer.from_pretrained(model_name)
        self.retriever = RagRetriever.from_pretrained(
            model_name,
            index_name="custom",
            passages_path=f"{retriever_index_path}/passages.json",
            index_path=f"{retriever_index_path}/index.faiss",
        )
        self.model = RagSequenceForGeneration.from_pretrained(model_name)

    def generate_answer(self, query: str, num_return_sequences: int = 1, num_beams: int = 2) -> str:
        """
        Generate an answer using the RAG pipeline based on the query.

        Args:
            query (str): The input question or query.
            num_return_sequences (int): Number of answers to return.
            num_beams (int): Beam search size.

        Returns:
            str: Generated answer(s).
        """
        # Tokenize the input query
        input_ids = self.tokenizer([query], return_tensors="pt", truncation=True, padding=True).input_ids

        # Use the retriever to find relevant passages
        retrieved_docs = self.retriever(input_ids.numpy(), return_tensors="pt")

        # Generate an answer using the retrieved documents
        outputs = self.model.generate(
            input_ids,
            context_input_ids=retrieved_docs["context_input_ids"],
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
        )

        # Decode and return the generated answers
        answers = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return answers
