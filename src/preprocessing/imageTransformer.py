import sys
from os import makedirs, path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from os import makedirs, path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from helpers.constants import PREPROCESSED_IMAGES_PATH
from helpers.data_load import load_image_files, stream_images


def preprocess_and_save_images_from_loader(train=True, test=True, validation=True, output_root=None, size=(448, 448)):
    """
    Preprocess images using the existing data loader and save them as tensors.

    Args:
        train (bool): Whether to preprocess training images.
        test (bool): Whether to preprocess test images.
        validation (bool): Whether to preprocess validation images.
        output_root (str): Root directory to save the preprocessed tensors.
    """

    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )

    image_splits = load_image_files(train=train, test=test, validation=validation)

    for split_name, image_paths in image_splits.items():
        output_dir = path.join(output_root, split_name)
        makedirs(output_dir, exist_ok=True)

        print(f"Processing {split_name} images...")

        for img_path, img in tqdm(
            zip(image_paths, stream_images(image_paths)), total=len(image_paths), desc=f"Preprocessing {split_name}"
        ):
            try:
                tensor = transform(img.convert("RGB"))
                filename = path.join(output_dir, f"{path.splitext(path.basename(img_path))[0]}.pt")
                torch.save(tensor, filename)
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")


if __name__ == "__main__":
    output_root = path.join(PREPROCESSED_IMAGES_PATH)
    preprocess_and_save_images_from_loader(train=True, test=True, validation=True, output_root=output_root)
