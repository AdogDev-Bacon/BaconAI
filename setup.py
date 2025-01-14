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
import setuptools

with open("requirements.txt", "r", encoding="utf-8") as f:
    req = f.readlines()
req = [x.strip() for x in req if x.strip()]

setuptools.setup(
    name="BaconAI-agents",
    version="1.0.0",
    author="ADogDev",
    author_email="",
    description="An Open-source AI agent that redefines how users interact with the Web3 ecosystem, provided by Bacon AI",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    packages=["agents"],
    python_requires=">=3.8",
    license="Apache License 2.0",
    install_requires=req,
)
