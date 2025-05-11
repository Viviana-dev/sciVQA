from PIL import ImageOps
from qwen_vl_utils.vision_process import process_vision_info


def custom_process_vision_info(messages, return_video_kwargs=False):
    image_inputs, video_inputs, *rest = process_vision_info(messages, return_video_kwargs=return_video_kwargs)

    if image_inputs is not None:
        padded_images = []
        for img in image_inputs:
            padding = int(0.1 * max(img.size))  # 10% padding
            padded_img = ImageOps.expand(img, border=padding, fill="white")
            padded_images.append(padded_img)
        image_inputs = padded_images

    return (image_inputs, video_inputs, *rest) if return_video_kwargs else (image_inputs, video_inputs)
