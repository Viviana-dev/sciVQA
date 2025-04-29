from pathlib import Path
from datasets import load_dataset, config
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from transformers import ViltProcessor, ViltForQuestionAnswering
import torch

# Load ViLT VQA model and processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Load BLIP VQA model and processor
# processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", use_fast=True)
# model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Load sample from dataset
config.DOWNLOADED_DATASETS_PATH = Path("/var/huggingface/datasets")

image_root = Path("./sciVQA/data/images/train/images_train") 

ds = load_dataset("katebor/SciVQA")
sample = ds["train"][57]

img_path = image_root / sample["image_file"]
question = sample["question"]

# Load and preprocess image
raw_image = Image.open(img_path).convert("RGB")
inputs = processor(raw_image, question, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits
predicted_idx = logits.argmax(-1)
answer = model.config.id2label[predicted_idx.item()]

# Inference
with torch.no_grad():
    out = model(**inputs)
    logits = out.logits
    predicted_idx = logits.argmax(-1)
    answer = model.config.id2label[predicted_idx.item()]

print(f"Q: {question}")
print(f"Predicted A: {answer}")
print(f"Ground Truth A: {sample['answer']}")
