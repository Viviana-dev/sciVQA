import sys
from os import path
from pathlib import Path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from evaluation.evaluation import evaluate_model_predictions
from evaluation.scoring import compute_evaluation_scores
from helpers.constants import LORA_PATH
from training.gpu_cleaner import clear_memory

# ---- Training Parameters ----
MODEL_NAME = "google/gemma-3-12b-it"
VERSION = "Version_17"

# ---- Clear Memory ----
clear_memory()

# ---- Evaluate the Model ----
evaluate_model_predictions(
    adapter_path=None,
    model_name=MODEL_NAME,
    version=VERSION,
    dataset_type="validation",
)

# ---- Calculate the Scores ----
compute_evaluation_scores(version=VERSION)
