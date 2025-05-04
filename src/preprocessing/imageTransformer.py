import torch
from PIL import Image
from transformers import AutoImageProcessor


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor: AutoImageProcessor):
    new_images = []
    for image in images:
        image = expand2square(image, tuple(int(1 * 255) for x in image_processor.image_mean))
        image = image_processor.preprocess(image, return_tensors="pt", size=image.size)["pixel_values"][0]
        new_images.append(image)

    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def get_image_processor(model_name, image_size):
    image_processor: AutoImageProcessor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
    if image_processor.size != image_size:
        image_processor.size = image_size
    return image_processor


def tensor_to_image(tensor, image_processor: AutoImageProcessor):
    """
    Converts a processed tensor back to a PIL image and displays it.
    """
    mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
    std = torch.tensor(image_processor.image_std).view(3, 1, 1)
    tensor = tensor * std + mean

    tensor = tensor.permute(1, 2, 0)  # Change shape from (C, H, W) to (H, W, C)
    tensor = (tensor * 255).clamp(0, 255).byte()  # Scale to [0, 255] and convert to uint8
    image = Image.fromarray(tensor.numpy())

    image.show()


if __name__ == "__main__":
    image_processor = get_image_processor("google/siglip2-base-patch16-512", 512)
    image = Image.open("data/images/validation/1811.05370v1-Figure3-1.png")
    processed_image = process_images([image], image_processor)
    print(processed_image.shape)

    tensor_to_image(processed_image[0], image_processor)
