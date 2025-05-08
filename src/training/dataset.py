import sys
from os import path
from typing import Literal

import pandas as pd
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from evaluation.evaluation import build_dynamic_prompt
from helpers.data_load import load_datasets, load_real_image_path


def convert_to_conversation(entry: pd.Series, split: str) -> list[dict[str, any]]:
    prompt_text, _ = build_dynamic_prompt(entry, split=split)

    # absolute or resolved file path to the image, so the Trainer/Processor
    # can find it even after changing working dirs
    image_path: str = load_real_image_path(entry["image_file"], **{split: True})
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": entry.get("answer", "")},
            ],
        },
    ]
    return conversation


class SciVQAConversationDataset(Dataset):

    def __init__(self, split: Literal["train", "validation", "test"] = "train") -> None:
        self.split = split
        if split not in ["train", "validation", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'validation', 'test']")

        if split == "train":
            self.table = load_datasets(train=True, test=False, validation=False)
        elif split == "validation":
            self.table = load_datasets(train=False, test=False, validation=True)
        elif split == "test":
            self.table = load_datasets(train=False, test=True, validation=False)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> dict[str, any]:
        entry = self.table.iloc[idx]
        dialog = convert_to_conversation(entry, split=self.split)
        return {
            "instance_id": entry["instance_id"],
            "messages": dialog,
        }
