import torch
import torchvision.transforms as T
from transformers import AutoModel, AutoProcessor

# Load a pre-trained CLIP model and processor
model_name = "openai/clip-vit-large-patch14"  # Or a SigLIP model name
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

import requests

# Load an image (e.g., using PIL)
from PIL import Image

image_path = "data/images/validation/1811.05370v1-Figure3-1.png"
image = Image.open(image_path).convert("RGB")
print(image.size)  # Check the image size

# Preprocess the image
inputs = processor(images=image, return_tensors="pt")
print(inputs["pixel_values"].shape)  # Check the inputs

# Get the image embeddings
with torch.no_grad():
    image_embeddings = model.vision_model(**inputs).pooler_output

print(image_embeddings.shape)  # Output shape will depend on the model

processed_tensor = inputs["pixel_values"][0]  # Shape: [3, H, W]

# Undo normalization if needed (CLIP uses specific mean/std)
mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]
unnormalize = T.Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std])
unnorm_img = unnormalize(processed_tensor).clamp(0, 1)

# Convert to PIL Image
to_pil = T.ToPILImage()
processed_image = to_pil(unnorm_img)

# Show the image
processed_image.show()
