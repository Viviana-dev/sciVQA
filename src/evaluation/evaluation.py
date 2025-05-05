import sys
from os import makedirs, path

import torch
import tqdm
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from helpers.constants import PREDICITION_PATH
from helpers.data_load import load_datasets, load_images, load_real_image_path


def evaluate_model(processor, model):
    validation_dataset = load_datasets(validation=True, train=False, test=False)
    total_rows = len(validation_dataset)

    results = []
    for i, entry in tqdm.tqdm(
        validation_dataset.iterrows(), total=total_rows, desc="Evaluating", unit="entry", unit_scale=True
    ):
        instance_id = entry["instance_id"]
        image_path = entry["image_file"]
        question = entry["question"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": load_real_image_path(image_path, validation=True)},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        answer = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        results.append({"instance_id": instance_id, "answer": answer})

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

    results = evaluate_model(processor, model)
    # save results to a csv file
    prediction_file_path = path.join(PREDICITION_PATH, "predictions", "predictions.csv")
    if not path.exists(path.dirname(prediction_file_path)):
        print(f"Creating directory: {path.dirname(prediction_file_path)}")
        makedirs(path.dirname(prediction_file_path))

    with open(prediction_file_path, "w") as f:
        f.write("instance_id,answer\n")
        for result in results:
            f.write(f"{result['instance_id']},{result['answer']}\n")
    print(f"Predictions saved to {prediction_file_path}")
