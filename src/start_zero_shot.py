import sys
from os import path

from accelerate import Accelerator

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from evaluation.evaluation import evaluate_model_predictions
from helpers.constants import LORA_PATH
from helpers.logging_utils import setup_logger

# ---- Zero-Shot Parameters ----
MODEL_NAME = "google/gemma-3-12b-it"
# MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
VERSION = "Version_25"
ACCELERATE = True


def main():
    accelerator = Accelerator() if ACCELERATE else None
    logger = setup_logger(__name__, accelerator=accelerator)

    logger.info(f"Using device: {accelerator.device}")
    logger.info("#" * 50)
    logger.info(f"Model Name: {MODEL_NAME}")
    logger.info(f"Version: {VERSION}")
    logger.info(f"Start Zero-Shot Evaluation!!")
    logger.info("#" * 50 + "\n\n")

    evaluate_model_predictions(
        adapter_path=None,
        model_name=MODEL_NAME,
        version=VERSION,
        dataset_type="validation",
        accelerate=True,
        scoring=True,
        dataset_name="scivqa",
        # dataset_name="chartqa",
        dynamic_prompt=True,
    )


if __name__ == "__main__":
    main()
