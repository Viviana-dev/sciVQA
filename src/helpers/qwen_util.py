from typing import Optional

import torch
from PIL import Image, ImageOps
from qwen_vl_utils.vision_process import process_vision_info


def custom_process_vision_info(
    messages: list[list[dict]] | list[dict], return_video_kwargs=False
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:
    """
    Custom wrapper for the `process_vision_info` function from `qwen_vl_utils` to add padding to images.

    Parameters
    ----------
    messages : list[list[dict]] | list[dict]
        The input messages containing image and video information.
    return_video_kwargs : bool
        If True, the function will return video-related information as well.

    Returns
    -------
    tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]
        A tuple containing the processed image inputs, video inputs, and any additional information.

    """
    image_inputs, video_inputs, *rest = process_vision_info(messages, return_video_kwargs=return_video_kwargs)

    if image_inputs is not None:
        padded_images = []
        for img in image_inputs:
            padding = int(0.1 * max(img.size))  # 10% padding
            padded_img = ImageOps.expand(img, border=padding, fill="white")
            padded_images.append(padded_img)
        image_inputs = padded_images

    return (image_inputs, video_inputs, *rest) if return_video_kwargs else (image_inputs, video_inputs)
