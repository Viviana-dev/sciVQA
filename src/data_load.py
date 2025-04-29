from pathlib import Path
from datasets import load_dataset, config
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch

# Login using e.g. `huggingface-cli login` to access this dataset
config.DOWNLOADED_DATASETS_PATH = Path("/var/huggingface/datasets")

ds = load_dataset("katebor/SciVQA")

image_root = Path("./sciVQA/data/images/train/images_train") 

sample = (ds["train"][1])

# Function to get image + QA pair
def get_sample(index=0):
    item = ds["train"][index]
    img_path = image_root / item['image_file']
    
    if img_path.exists():
        img = Image.open(img_path)
        print(f"Q: {item['question']}")
        print(f"A: {item['answer']}")
        img.show()
    else:
        print(f"Image not found: {img_path}")

# Example usage
get_sample(0)


# Load BLIP VQA model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Load sample from dataset
sample = dataset[0]
img_path = image_root / sample["image_file"]
question = sample["question"]

# Load and preprocess image
raw_image = Image.open(img_path).convert("RGB")
inputs = processor(raw_image, question, return_tensors="pt")

# Inference
with torch.no_grad():
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)

print(f"Q: {question}")
print(f"Predicted A: {answer}")
print(f"Ground Truth A: {sample['answer']}")

