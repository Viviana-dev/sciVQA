from os import path
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent.parent
SRC_PATH = path.join(BASE_PATH, "src")
DATA_PATH = path.join(BASE_PATH, "data")
PREPROCESSED_IMAGES_PATH = path.join(DATA_PATH, "preprocessed")
IMAGES_PATH = path.join(DATA_PATH, "images")
CSV_PATH = path.join(DATA_PATH, "csv")
PREDICITION_PATH = path.join(BASE_PATH, "predictions")
OUTPUT_PATH = path.join(BASE_PATH, "output")
LORA_PATH = path.join(OUTPUT_PATH, "lora")
