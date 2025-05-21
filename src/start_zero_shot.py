import sys
from os import makedirs, path
from pathlib import Path

import torch
from accelerate import Accelerator

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from evaluation.evaluation import evaluate_model_predictions
from helpers.constants import LORA_PATH
from helpers.logging_utils import setup_logger
from training.finetuning import trainLoraModel
from training.gpu_cleaner import clear_memory

# ---- Training Parameters ----

MODEL_NAME = "google/gemma-3-12b-it"  # "Qwen/Qwen2.5-VL-7B-Instruct"
VERSION = "Version_25"
OUTPUT_DIR = Path(path.join(LORA_PATH, "no-ocr-v4", VERSION))
if not OUTPUT_DIR.exists():
    makedirs(OUTPUT_DIR, exist_ok=True)
BATCH_SIZE = 6
GRAD_ACC = 4
EPOCHS = 2
LR = 2e-4
DTYPE = torch.bfloat16
LORA_RANK = 64
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = r"^(?!.*visual).*(?:o_proj|up_proj|v_proj|down_proj|k_proj|q_proj|gate_proj).*"
ACCELERATE = True


def main():
    accelerator = Accelerator() if ACCELERATE else None
    logger = setup_logger(__name__, accelerator=accelerator)

    logger.info(f"Using device: {accelerator.device}")
    logger.info("#" * 50)
    logger.info(f"Model Name: {MODEL_NAME}")
    logger.info(f"Version: {VERSION}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Target Modules: {TARGET_MODULES}")
    logger.info("#" * 50 + "\n\n")

    logger.info("Training completed. Clearing memory...")
    if accelerator:
        # ---- Clear Memory ----
        if accelerator.is_local_main_process:
            clear_memory(accelerator)
    else:
        clear_memory()
    logger.info("Memory cleared.")

    # ---- Evaluate the Model ----
    evaluate_model_predictions(
        adapter_path=None,
        model_name=MODEL_NAME,
        version="Version_25",
        dataset_type="validation",
        accelerate=True,
        scoring=True,
        dataset_name="scivqa",
        # dataset_name="chartqa",
    )


if __name__ == "__main__":
    main()
