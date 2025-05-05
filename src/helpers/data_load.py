import sys
from email.mime import base
from os import listdir, makedirs, path, remove, rename

from PIL import Image

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import shutil
from typing import Generator, Literal
from zipfile import ZipFile

import pandas as pd
import requests

from helpers.constants import CSV_PATH, DATA_PATH, IMAGES_PATH


def downlaode_csv(train: bool = True, test: bool = True, validation: bool = True):
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
                print(f"Creating directory: {DATA_PATH}")
                makedirs(DATA_PATH)
            if not path.exists(path.join(CSV_PATH)):
                print(f"Creating directory: {path.join(CSV_PATH)}")
                makedirs(path.join(CSV_PATH))
            print(f"Downloading {split_name} dataset...")
            csv_path = path.join(CSV_PATH, f"{split_name}.csv")

            js_data = pd.read_json(json_path)
            js_data.to_csv(csv_path, index=False)


def downlaode_images(train: bool = True, test: bool = True, validation: bool = True):
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
                print(f"{split_name} images already downloaded.")
                continue

            zip_path = path.join(IMAGES_PATH, f"{split_name}.zip")
            if not path.exists(path.join(IMAGES_PATH)):
                print(f"Creating directory: {path.join(DATA_PATH, 'images')}")
                makedirs(path.join(IMAGES_PATH))
            print(f"Downloading {split_name} images...")
            if not path.exists(zip_path):
                response = requests.get(files[split_name])
                with open(zip_path, "wb") as f:
                    f.write(response.content)
            print(f"Unzipping {split_name} images...")

            with ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(path.join(IMAGES_PATH))

            for unzipped_folders in listdir(path.join(IMAGES_PATH)):
                if unzipped_folders.startswith("images_"):
                    rename(
                        path.join(IMAGES_PATH, unzipped_folders),
                        path.join(IMAGES_PATH, split_name),
                    )
                    print(f"Renamed {unzipped_folders} to {split_name}")
                    break

            print(f"Deleting {split_name} zip file...")
            remove(zip_path)

    macosx_folder = path.join(IMAGES_PATH, "__MACOSX")
    if path.exists(macosx_folder):
        shutil.rmtree(macosx_folder)


def load_datasets(
    train: bool = True, test: bool = True, validation: bool = True
) -> dict[Literal["train", "test", "validation"], pd.DataFrame]:
    """
    Load the datasets from the csv files.
    If the csv files do not exist, download them from the internet.
    Args:
        train (bool): If True, load the train dataset.
        test (bool): If True, load the test dataset.
        validation (bool): If True, load the validation dataset.
    Returns:
        dict: A dictionary with the datasets.
            The keys are the names of the datasets (train, test, validation).
            The values are the datasets as pandas DataFrames.
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


def load_image_files(
    train: bool = True, test: bool = True, validation: bool = True
) -> dict[Literal["train", "test", "validation"], list[str]]:
    """
    Load the image files from the dataset.
    If the image files do not exist, download them from the internet.
    Args:
        train (bool): If True, load the train images.
        test (bool): If True, load the test images.
        validation (bool): If True, load the validation images.
    Returns:
        dict: A dictionary with the image files.
            The keys are the names of the datasets (train, test, validation).
            The values are the image files as lists of strings.
    """

    def get_images(split_name):
        images_path = path.join(IMAGES_PATH, split_name)
        if not path.exists(images_path):
            downlaode_images(**{split_name: True})
        return [path.join(images_path, f) for f in listdir(images_path) if f.endswith(".png")]

    splits = {"train": train, "test": test, "validation": validation}
    images = {name: get_images(name) for name, flag in splits.items() if flag}

    return images


def stream_images(file_paths: list[str]) -> Generator[Image.Image, None, None]:
    """
    Yield image data from a list of file paths.
    Args:
        file_paths (list[str]): List of image file paths.
    Yields:
        bytes: The image data.
    """
    for file_path in file_paths:
        with Image.open(file_path) as img:
            yield img.copy()


def load_images(dataset_path: str, train: bool = False, test: bool = False, validation: bool = True) -> Image.Image:
    """
    Load the image based ont the file name provided by the dataset path.
    Args:
        dataset_path (str): The path to the dataset.
        train (bool): If True, load the train images.
        test (bool): If True, load the test images.
        validation (bool): If True, load the validation images.
    Returns:
        the Image object.
    """
    splits = {"train": train, "test": test, "validation": validation}
    selected_splits = {name: flag for name, flag in splits.items() if flag}
    if len(selected_splits) != 1:
        raise ValueError("Please provide only one of train, test or validation")
    split_name = next(iter(selected_splits.keys()))
    images_path = path.join(IMAGES_PATH, split_name)
    if not path.exists(images_path):
        print(f"Image sub folder {split_name} path {images_path} does not exist")
        downlaode_images()

    image_path = path.join(images_path, dataset_path)
    if not path.exists(image_path):
        raise ValueError(f"Image path {image_path} does not exist")
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        return img.copy()


def load_real_image_path(dataset_path: str, train: bool = False, test: bool = False, validation: bool = True) -> str:
    """
    Load the image based ont the file name provided by the dataset path.
    Args:
        dataset_path (str): The path to the dataset.
        train (bool): If True, load the train images.
        test (bool): If True, load the test images.
        validation (bool): If True, load the validation images.
    Returns:
        the Image object.
    """
    splits = {"train": train, "test": test, "validation": validation}
    selected_splits = {name: flag for name, flag in splits.items() if flag}
    if len(selected_splits) != 1:
        raise ValueError("Please provide only one of train, test or validation")
    split_name = next(iter(selected_splits.keys()))
    images_path = path.join(IMAGES_PATH, split_name)
    if not path.exists(images_path):
        print(f"Image sub folder {split_name} path {images_path} does not exist")
        downlaode_images()

    image_path = path.join(images_path, dataset_path)
    if not path.exists(image_path):
        raise ValueError(f"Image path {image_path} does not exist")
    return image_path
