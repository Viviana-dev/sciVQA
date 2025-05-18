import sys
from os import makedirs, path
from pathlib import Path

import torch
from accelerate import Accelerator

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from helpers.constants import LORA_PATH
from training.finetuning import trainLoraModel

# ---- Training Parameters ----

MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"
VERSION = "Version_22"
OUTPUT_DIR = Path(path.join(LORA_PATH, "no-ocr-v4", VERSION))
if not OUTPUT_DIR.exists():
    makedirs(OUTPUT_DIR, exist_ok=True)
BATCH_SIZE = 4  # Batch size per device: This is the number of samples processed before the model is updated. A larger batch size can lead to better convergence but requires more memory.
GRAD_ACC = 2  # Gradient Accumulation Steps: This is used to simulate a larger batch size by accumulating gradients over multiple steps before updating the model weights.
EPOCHS = 2
LR = 2e-4
DTYPE = torch.bfloat16  # Use bfloat16 for training
LORA_RANK = (
    64  # Lora rank is the number of low-rank matrices to be used in the LoRA module: higher rank means more parameters
)
LORA_ALPHA = 32  # Lora alpha is the scaling factor for the low-rank matrices: higher alpha means more parameters
LORA_DROPOUT = (
    0.05  # Lora dropout is the dropout rate for the low-rank matrices: higher dropout means more regularization
)
TARGET_MODULES = "all-linear"

if Accelerator().is_main_process:
    print("#" * 20)
    print("Training LoRA Model")
    print(f"Model Name: {MODEL_NAME}")
    print(f"Version: {VERSION}")
    print(f"Target Modules: {TARGET_MODULES}")
    print("#" * 20)


# ---- Start Training ----
trainLoraModel(
    model_name=MODEL_NAME,
    version=VERSION,
    output_dir=OUTPUT_DIR,
    batch_size=BATCH_SIZE,
    grad_acc=GRAD_ACC,
    epochs=EPOCHS,
    lr=LR,
    dtype=DTYPE,
    lora_rank=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    accelerate=False,
)

# ---- Clear Memory ----
# clear_memory()

# ---- Evaluate the Model ----
# evaluate_model_predictions(
#    adapter_path=Path(path.join(OUTPUT_DIR, "model")),
#    model_name=MODEL_NAME,
#    version=VERSION,
#    dataset_type="validation",
# )

# ---- Calculate the Scores ----
# compute_evaluation_scores(version=VERSION)
