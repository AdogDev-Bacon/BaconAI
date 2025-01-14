import os
from agents import CryptoInfoDataset
import litellm
from agents.optimization.trainer import Trainer, TrainerConfig

#
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""

litellm.set_verbose = False

if __name__ == "__main__":
    dataset = CryptoInfoDataset(split="train")
    trainer_config_path = ""        # training config directory
    trainer = Trainer(config=TrainerConfig(
        trainer_config_path), dataset=dataset)
    trainer.train()
