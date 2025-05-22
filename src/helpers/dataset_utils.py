import shutil
import sys
from os import listdir, makedirs, path, remove, rename
from typing import Literal
from zipfile import ZipFile

import pandas as pd
import requests
from datasets import load_dataset
from torch.utils.data import Dataset

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from helpers.constants import CSV_PATH, DATA_PATH, IMAGES_PATH
from helpers.logging_utils import setup_logger
from helpers.prompt_utils import build_dynamic_prompt, build_general_prompt

logger = setup_logger(__name__)


def convert_to_conversation(
    entry: pd.Series, split: str, add_answer: bool = False, dynamic: bool = False
) -> list[dict[str, any]]:
    """
    Convert a dataset entry into a conversation format for the Vision Language Model.

    Parameters
    ----------
    entry : pd.Series
        A row from the dataset containing fields like 'image_file' and possibly 'answer'.
    split : str
        The data split ('train', 'validation', or 'test') indicating which images to load.
    add_answer : bool, default False
        Whether to include the correct answer in the conversation.
    dynamic : bool, default False
        Whether to use dynamic prompts for the dataset, or the more general one, which is compatible for other datasets.

    Returns
    -------
    list[dict[str, any]]
        A list of dictionaries representing the conversation messages for system, user, and optionally assistant.
    """
    if dynamic:
        prompt_text = build_dynamic_prompt(entry)
    else:
        prompt_text = build_general_prompt(entry)
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
    ]
    if add_answer:
        conversation.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": entry.get("answer", ""),
                    },
                ],
            },
        )
    return conversation


class SciVQAEvaluationDataset(Dataset):

    def __init__(self, split: Literal["train", "validation", "test"] = "train", dynamic: bool = False) -> None:
        """
        Initialize the evaluation dataset for SciVQA.

        Parameters
        ----------
        split : Literal["train", "validation", "test"], default "train"
            The dataset split to load.
        dynamic : bool, default False
            Whether to use dynamic prompts for the dataset, or the more general one, which is compatible for other datasets.

        Raises
        ------
        ValueError
            If split is not one of 'train', 'validation', or 'test'.
        """
        self.dynamic = dynamic
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
        """
        Return the number of examples in the dataset.

        Returns
        -------
        int
            The total number of entries loaded for the given split.
        """
        return len(self.table)

    def __getitem__(self, idx: int) -> dict[str, any]:
        """
        Get a single example by index from the evaluation dataset.

        Parameters
        ----------
        idx : int
            The index of the example to retrieve.

        Returns
        -------
        dict[str, any]
            A dictionary with keys 'instance_id', 'messages', and 'answer'.
        """
        entry = self.table.iloc[idx]
        dialog = convert_to_conversation(entry, split=self.split, add_answer=False, dynamic=self.dynamic)
        return {
            "instance_id": entry["instance_id"],
            "messages": dialog,
            "answer": entry.get("answer", ""),
        }

    def collate_fn(self, batch: list[dict[str, any]]) -> dict[str, any]:
        """
        Collate a batch of examples into batched lists.

        Parameters
        ----------
        batch : list[dict[str, any]]
            A list of dataset items each containing 'instance_id', 'messages', and 'answer'.

        Returns
        -------
        dict[str, any]
            A dictionary with batched 'instance_id', 'messages', and 'answer'.
        """
        return {
            "instance_id": [item["instance_id"] for item in batch],
            "messages": [item["messages"] for item in batch],
            "answer": [item["answer"] for item in batch],
        }


def convert_to_conversation_chartqa(entry: pd.Series, split: str, add_answer: bool = False) -> list[dict[str, any]]:
    """
    Convert a dataset entry into a conversation format for the Vision Language Model.

    Parameters
    ----------
    entry : pd.Series
        A row from the dataset containing fields like 'image_file' and possibly 'answer'.
    split : str
        The data split ('train', 'validation', or 'test') indicating which images to load.
    add_answer : bool, default False
        Whether to include the correct answer in the conversation.

    Returns
    -------
    list[dict[str, any]]
        A list of dictionaries representing the conversation messages for system, user, and optionally assistant.
    """
    prompt_text = build_general_prompt(entry)

    system_message: str = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

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
                    "image": entry["image_file"],
                },
                {
                    "type": "text",
                    "text": prompt_text,
                },
            ],
        },
    ]
    if add_answer:
        conversation.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": entry.get("answer", ""),
                    },
                ],
            },
        )
    return conversation


class ChartQAEvaluationDataset(Dataset):
    def __init__(self, split: Literal["train", "validation", "test"] = "train") -> None:
        """
        Initialize the evaluation dataset for ChartQA.

        Parameters
        ----------
        split : Literal["train", "validation", "test"], default "train"
            The dataset split to load.

        Raises
        ------
        ValueError
            If split is not one of 'train', 'validation', or 'test'.
        """
        self.split = split
        if split not in ["train", "validation", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'validation', 'test']")
        if split == "validation":
            self.split = "val"
            split = "val"

        ds = load_dataset("HuggingFaceM4/ChartQA", split=split)
        self.table = pd.DataFrame(ds)
        # rename the columns to match the SciVQA dataset
        self.table.rename(columns={"image": "image_file", "query": "question", "label": "answer"}, inplace=True)
        # add the instance_id column with the row index of the dataframe
        self.table["instance_id"] = self.table.index.to_list()
        # convert type of instance_id to string
        self.table["instance_id"] = self.table["instance_id"].astype(str)
        # convert the answer from list to string if it is a one element list
        self.table["answer"] = self.table["answer"].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> dict[str, any]:
        entry = self.table.iloc[idx]
        dialog = convert_to_conversation_chartqa(entry, split=self.split, add_answer=False)
        return {
            "instance_id": entry["instance_id"],
            "messages": dialog,
            "answer": entry.get("answer", ""),
        }

    def collate_fn(self, batch: list[dict[str, any]]) -> dict[str, any]:
        """
        Collate a batch of examples into batched lists.

        Parameters
        ----------
        batch : list[dict[str, any]]
            A list of dataset items each containing 'instance_id', 'messages', and 'answer'.

        Returns
        -------
        dict[str, any]
            A dictionary with batched 'instance_id', 'messages', and 'answer'.
        """
        return {
            "instance_id": [item["instance_id"] for item in batch],
            "messages": [item["messages"] for item in batch],
            "answer": [item["answer"] for item in batch],
        }


class SciVQATrainingDataset(Dataset):
    def __init__(self, split: Literal["train", "validation", "test"] = "train", dynamic: bool = False) -> None:
        """
        Initialize the training dataset for SciVQA.

        Parameters
        ----------
        split : Literal["train", "validation", "test"], default "train"
            The dataset split to load.
        dynamic : bool, default False
            Whether to use dynamic prompts for the dataset, or the more general one, which is compatible for other datasets.

        Raises
        ------
        ValueError
            If split is not one of 'train', 'validation', or 'test'.
        """
        self.dynamic = dynamic
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
        dialog = convert_to_conversation(entry, split=self.split, add_answer=True, dynamic=self.dynamic)
        return {
            "instance_id": entry["instance_id"],
            "messages": dialog,
        }


def downlaode_csv(train: bool = True, test: bool = True, validation: bool = True):
    """
    Download CSV splits for the SciVQA dataset and save them locally.

    Parameters
    ----------
    train : bool, default True
        Whether to download the training split.
    test : bool, default True
        Whether to download the test split.
    validation : bool, default True
        Whether to download the validation split.
    """
    splits = {
        "train": "train_2025-03-27_18-34-44.json",
        "validation": "validation_2025-03-27_18-34-44.json",
        "test": "test_without_answers_2025-04-14_15-30.json",
    }

    requested_splits = {"train": train, "test": test, "validation": validation}

    for split_name, flag in requested_splits.items():
        if flag:
            json_path = f"hf://datasets/katebor/SciVQA/{splits[split_name]}"
            if not path.exists(DATA_PATH):
                logger.info(f"Creating directory: {DATA_PATH}")
                makedirs(DATA_PATH)
            if not path.exists(path.join(CSV_PATH)):
                logger.info(f"Creating directory: {path.join(CSV_PATH)}")
                makedirs(path.join(CSV_PATH))
            logger.info(f"Downloading {split_name} dataset...")
            csv_path = path.join(CSV_PATH, f"{split_name}.csv")

            js_data = pd.read_json(json_path)
            js_data.to_csv(csv_path, index=False)


def downlaode_images(train: bool = True, test: bool = True, validation: bool = True):
    """
    Download and unzip image archives for the SciVQA dataset.

    Parameters
    ----------
    train : bool, default True
        Whether to download the training images.
    test : bool, default True
        Whether to download the test images.
    validation : bool, default True
        Whether to download the validation images.
    """
    files: dict[str, str] = {
        "test": "https://huggingface.co/datasets/katebor/SciVQA/resolve/main/images_test.zip",
        "train": "https://huggingface.co/datasets/katebor/SciVQA/resolve/main/images_train.zip",
        "validation": "https://huggingface.co/datasets/katebor/SciVQA/resolve/main/images_validation.zip",
    }

    requested_files = {"train": train, "test": test, "validation": validation}

    for split_name, flag in requested_files.items():
        if flag:
            # check if the directory with the unzipped images exists, if it exist skip downlaoding
            if path.exists(path.join(IMAGES_PATH, split_name)):
                logger.info(f"{split_name} images already downloaded.")
                continue

            zip_path = path.join(IMAGES_PATH, f"{split_name}.zip")
            if not path.exists(path.join(IMAGES_PATH)):
                logger.info(f"Creating directory: {path.join(DATA_PATH, 'images')}")
                makedirs(path.join(IMAGES_PATH))
            logger.info(f"Downloading {split_name} images...")
            if not path.exists(zip_path):
                response = requests.get(files[split_name])
                with open(zip_path, "wb") as f:
                    f.write(response.content)
            logger.info(f"Unzipping {split_name} images...")

            with ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(path.join(IMAGES_PATH))

            for unzipped_folders in listdir(path.join(IMAGES_PATH)):
                if unzipped_folders.startswith("images_"):
                    rename(
                        path.join(IMAGES_PATH, unzipped_folders),
                        path.join(IMAGES_PATH, split_name),
                    )
                    logger.info(f"Renamed {unzipped_folders} to {split_name}")
                    break

            logger.info(f"Deleting {split_name} zip file...")
            remove(zip_path)

    macosx_folder = path.join(IMAGES_PATH, "__MACOSX")
    if path.exists(macosx_folder):
        shutil.rmtree(macosx_folder)


def load_datasets(
    train: bool = True, test: bool = True, validation: bool = True
) -> dict[Literal["train", "test", "validation"], pd.DataFrame] | pd.DataFrame:
    """
    Load dataset CSV files into pandas DataFrames, downloading if necessary.

    Parameters
    ----------
    train : bool, default True
        Whether to load the training split.
    test : bool, default True
        Whether to load the test split.
    validation : bool, default True
        Whether to load the validation split.

    Returns
    -------
    pd.DataFrame or dict[str, pd.DataFrame]
        A single DataFrame if only one split is requested; otherwise a dict mapping split names to DataFrames.
    """

    def get_dataset(split_name):
        csv_path = path.join(CSV_PATH, f"{split_name}.csv")
        if not path.exists(csv_path):
            downlaode_csv(**{split_name: True})
        return pd.read_csv(csv_path)

    splits = {"train": train, "test": test, "validation": validation}
    selected_splits = {name: flag for name, flag in splits.items() if flag}

    if len(selected_splits) == 1:
        split_name = next(iter(selected_splits.keys()))
        return get_dataset(split_name)

    datasets = {name: get_dataset(name) for name, flag in splits.items() if flag}

    return datasets


def load_real_image_path(dataset_path: str, train: bool = False, test: bool = False, validation: bool = False) -> str:
    """
    Get the file path for an image in the specified dataset split.

    Parameters
    ----------
    dataset_path : str
        The filename of the image within the split directory.
    train : bool, default False
        Get path from the training split.
    test : bool, default False
        Get path from the test split.
    validation : bool, default False
        Get path from the validation split.

    Returns
    -------
    str
        The full filesystem path to the image file.

    Raises
    ------
    ValueError
        If none or multiple splits are specified, or if the image file does not exist.
    """
    splits = {"train": train, "test": test, "validation": validation}
    selected_splits = {name: flag for name, flag in splits.items() if flag}
    if len(selected_splits) != 1:
        raise ValueError("Please provide only one of train, test or validation")
    split_name = next(iter(selected_splits.keys()))
    images_path = path.join(IMAGES_PATH, split_name)
    if not path.exists(images_path):
        logger.info(f"Image sub folder {split_name} path {images_path} does not exist")
        downlaode_images()

    image_path = path.join(images_path, dataset_path)
    if not path.exists(image_path):
        raise ValueError(f"Image path {image_path} does not exist")
    return image_path


if __name__ == "__main__":
    dataset = ChartQAEvaluationDataset(split="validation")
    print(dataset[0])
