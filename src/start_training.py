import sys
from os import makedirs, path
from pathlib import Path

import torch

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from evaluation.evaluation import evaluate_model_predictions
from evaluation.scoring import compute_evaluation_scores
from helpers.constants import LORA_PATH
from training.gpu_cleaner import clear_memory
from training.qwen.finetuning import trainLoraModel

# ---- Training Parameters ----

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
VERSION = "Version_9"
OUTPUT_DIR = Path(path.join(LORA_PATH, "no-ocr-v4", VERSION))
if not OUTPUT_DIR.exists():
    makedirs(OUTPUT_DIR, exist_ok=True)
BATCH_SIZE = 12  # Batch size per device: This is the number of samples processed before the model is updated. A larger batch size can lead to better convergence but requires more memory.
GRAD_ACC = 2  # Gradient Accumulation Steps: This is used to simulate a larger batch size by accumulating gradients over multiple steps before updating the model weights.
EPOCHS = 5
LR = 2e-4
COMPUTE_DTYPE = torch.bfloat16  # Use bfloat16 for training
LORA_RANK = (
    64  # Lora rank is the number of low-rank matrices to be used in the LoRA module: higher rank means more parameters
)
LORA_ALPHA = 32  # Lora alpha is the scaling factor for the low-rank matrices: higher alpha means more parameters
LORA_DROPOUT = (
    0.05  # Lora dropout is the dropout rate for the low-rank matrices: higher dropout means more regularization
)
TARGET_MODULES = [
    "up_proj",
    "gate_proj",
    "down_proj",
    "q_proj",
    "v_proj",
    "visual.blocks.X.attn.qkv",
    "visual.blocks.X.attn.proj",
]  # Target modules for LoRA

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
    compute_dtype=COMPUTE_DTYPE,
    lora_rank=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
)

# ---- Clear Memory ----
clear_memory()

# ---- Evaluate the Model ----
evaluate_model_predictions(
    adapter_path=Path(path.join(OUTPUT_DIR, "model")),
    model_name=MODEL_NAME,
    version=VERSION,
)

# ---- Calculate the Scores ----
compute_evaluation_scores(version=VERSION)
