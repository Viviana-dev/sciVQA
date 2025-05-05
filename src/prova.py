import sys
from os import path
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from preprocessing.imageTransformer import expand2square

# Load ViLT VQA model and processor
model = PaliGemmaForConditionalGeneration.from_pretrained("ahmed-masry/chartgemma", torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma", use_fast=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_root = Path("data/images/train")

ds = load_dataset("katebor/SciVQA")
sample = ds["train"][0]

img_path = image_root / sample["image_file"]
question = sample["question"]
print(f"Question: {question}")

input_prompt = f"<image>\n Question: {question} Answer: "

# Load and preprocess image
raw_image = Image.open(img_path).convert("RGB")
raw_image.show()
raw_image = expand2square(raw_image, tuple([255, 255, 255]))
inputs = processor(images=raw_image, text=input_prompt, return_tensors="pt")
inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
inputs = {k: v.to(device) for k, v in inputs.items()}
prompt_length = inputs["input_ids"].shape[1]


generate_ids = model.generate(**inputs, num_beams=4, max_new_tokens=512)
output_text = processor.batch_decode(
    generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(f"Q: {question}")
print(f"Predicted A: {output_text}")
print(f"Ground Truth A: {sample['answer']}")
