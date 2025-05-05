import torch
from PIL import Image
from transformers import AutoImageProcessor, Siglip2ImageProcessor


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


def process_images(images, image_processor: Siglip2ImageProcessor):
    new_images = []
    for image in images:
        # image = expand2square(image, tuple([255, 255, 255]))
        image = image_processor.preprocess(image, return_tensors="pt", do_resize=False, size=(512, 512))[
            "pixel_values"
        ][0]
        new_images.append(image)

    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def get_image_processor(model_name, image_size):
    image_processor: Siglip2ImageProcessor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
    if image_processor.size != image_size:
        image_processor.size = image_size
    return image_processor


if __name__ == "__main__":
    image_processor = get_image_processor("google/siglip2-base-patch16-512", 512)
    image = Image.open("data/images/validation/1811.05370v1-Figure3-1.png")
    processed_image = process_images([image], image_processor)
    print(processed_image.shape)
