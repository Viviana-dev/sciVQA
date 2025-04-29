from pathlib import Path
from os import path

BASE_PATH = Path(__file__).resolve().parent.parent.parent
SRC_PATH = path.join(BASE_PATH, "src")
DATA_PATH = path.join(BASE_PATH, "data")