from os import path
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent.parent
SRC_PATH = path.join(BASE_PATH, "src")
DATA_PATH = path.join(BASE_PATH, "data")
PREDICITION_PATH = path.join(BASE_PATH, "predictions")
SCORES_PATH = path.join(BASE_PATH, "Scores_versions")
METRIC_PATH = path.join(BASE_PATH, "Metrics_versions")
LOARA_VERSIONS_PATH = path.join(BASE_PATH, "LoRa_versions")
PREPROCESSED_IMAGES_PATH = path.join(DATA_PATH, "preprocessed")
IMAGES_PATH = path.join(DATA_PATH, "images")
CSV_PATH = path.join(DATA_PATH, "csv")
OUTPUT_PATH = path.join("/", "var", "sciVQA", "output")
LORA_PATH = path.join(OUTPUT_PATH, "lora")

SPECIAL_TOKENS = {"additional_special_tokens": ["<thinking>", "<answer>"]}
