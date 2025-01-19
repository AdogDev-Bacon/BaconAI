<div align="center">

## <p align="center"><img src='./figures/logo.png' width="28" height="26"> Bacon AI agent </p>

<p align="center">

[//]: # (<a href="">[🤖Project]</a>)

[//]: # (<a href="">[📝Docs]</a>)

[//]: # (<a href="#overview">[🌟Overview]</a>)
</p>

![](https://img.shields.io/badge/version-v1.0.0-yellow)
[![License: Apache](https://img.shields.io/badge/License-Apache2.0-blue.svg)](https://opensource.org/license/apache-2-0)
![](https://img.shields.io/github/last-commit/aiwaves-cn/agents?color=green)


<img src='./figures/banner.png' width="900" height="300">

---

</div>


## News
- [2025-01-13] We proudly release Bacon AI agent 1.0!
---

## Overview

Bacon AI is a groundbreaking AI agent that redefines how users interact with the Web3 ecosystem. Built on cutting-edge technology and seamlessly integrated with multiple secure platforms, Bacon AI empowers users to handle complex tasks with ease and precision.

### A One-for-All Solution Through Seamless Integration
Bacon AI revolutionizes the Web3 user experience by acting as a one-for-all solution that seamlessly integrates with a wide range of platforms. By consolidating the functionalities of crypto exchange platforms, crypto wallets, and blockchain analysis tools into a single AI agent, Bacon AI eliminates the need for multiple applications. Users can perform token trades, manage multi-wallet operations, monitor real-time market trends, and execute transactions directly through the agent, without switching between different apps. This unified approach streamlines workflows, reduces complexity, and empowers users to manage all their Web3 activities effortlessly. Bacon AI's ability to interact with diverse platforms makes it an indispensable tool for anyone navigating the decentralized world.

<img src='./figures/Bacon_AI_2.0.drawio.png'>


### Retrieval-Augmented Generation Pipeline

At the heart of the agent is a Retrieval-Augmented Generation (RAG) pipeline, which seamlessly integrates the LLMs with a high-performance vector database. This infrastructure indexes diverse datasets, such as real-time token performance, historical trends, and social media sentiment, enabling the agent to:
-   Deliver instant and contextually relevant responses for blockchain-related queries.
- Provide granular insights into liquidity optimization, cross-chain operations, and market dynamics.
- Enhance decision-making with real-time data synchronization and contextual analysis.
The RAG pipeline synergizes neural inference with vector search to exceed the capabilities of traditional blockchain platforms, providing a highly granular and precise understanding of complex datasets. Furthermore, Bacon AI utilizes reinforcement learning to mimic real human analysis of token markets. It receives rewards based on how closely its predictions align with actual market trends and established influencer analyses. With reinforcement learning, Bacon AI is empowered to generate reasonable, actionable insights that support the Web 3 community, ensuring its outputs remain both relevant and impactful.


<img src='./figures/Bacon_AI_RAG.drawio.png'>
---
### Industry-leadning Security and Precision 

Bacon AI sets a new standard for secure and accurate AI-driven interactions. It employs:
-   External Safety Filters: These filters actively detect and prevent harmful or unethical outputs, ensuring users receive reliable and safe responses.
-   Auxiliary Models: These models provide a second layer of validation, mitigating hallucinations and enhancing the precision of Bacon AI's outputs.
These mechanisms are especially vital in the Web3 ecosystem, where malicious users could exploit AI agents for scams, phishing, or other fraudulent activities, potentially compromising the integrity of the community. By proactively addressing these risks, Bacon AI delivers a secure and trustworthy Q&A experience tailored to the complexities of Web3. 

<img src='./figures/Bacon_AI_safety.drawio.png'>

---
## Installation
```
git clone https://github.com/AdogDev-Bacon/BaconAI
cd BaconAI
pip install -e .
```

---
### Building your own AI agent

#### Quick Start

Users can customize their own AI agents by first providing config files and fine-tune data under corresponding directories.
```
|-- examples/
|   |-- chatbot/
|   |   |--configs.json
|   |-- cryto_info/
|   |   |--configs.json
|-- src/
|   |-- agents/
|   |   |--datasets/
|   |   |   |--data/
        
```

Then in run_train.py, users can add their OpenAI API key and dataset configs to fine-tune a customized AI agent. 

```
import os
from agents import PREDEFINED_DATASET
import litellm
from agents.optimization.trainer import Trainer, TrainerConfig

os.environ["OPENAI_API_KEY"] = OPENAI_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL

litellm.set_verbose = False

if __name__ == "__main__":
    dataset = PREDEFINED_DATASET(split="train")
    trainer_config_path = CONFIG_DIRECTORY
    trainer = Trainer(config=TrainerConfig(
        trainer_config_path), dataset=dataset)
    trainer.train()
```

---
### To-do
- Enable the agent to access and perform operations on crypto wallets.
- Allow the agent to interact with and execute tasks on crypto trading platforms.
---

### Paper
Our paper on Bacon AI agent can be found at here [📄Paper](files/Bacon_AI-2.pdf)






