import sys
from os import path
from typing import Literal

import pandas as pd
from torch.utils.data import Dataset

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from evaluation.evaluation import build_dynamic_prompt
from helpers.data_load import load_datasets, load_real_image_path


def convert_to_conversation(entry: pd.Series, split: str) -> list[dict[str, any]]:
    prompt_text, _ = build_dynamic_prompt(entry, split=split)

    system_message: str = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

    image_path: str = load_real_image_path(entry["image_file"], **{split: True})
    conversation: list[dict[str, any]] = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": prompt_text,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": entry.get("answer", ""),
                },
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
