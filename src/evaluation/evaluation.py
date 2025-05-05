import ast
import csv
import sys
from os import makedirs, path

import pandas as pd
import torch
import tqdm
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from helpers.constants import PREDICITION_PATH
from helpers.data_load import load_datasets, load_real_image_path


def build_dynamic_prompt(entry, split="validation"):
    question = entry["question"]
    qa_type = entry["qa_pair_type"]
    caption = entry.get("caption", "")
    figure_type = entry.get("figure_type", "figure")
    compound = entry.get("compound", False)
    figs_numb = entry.get("figs_numb", 0)
    answer_options = entry.get("answer_options", "")

    prompt = f"You are looking at a {figure_type}"
    if compound:
        prompt += f" with {figs_numb} subfigures"
    prompt += "."

    if caption:
        prompt += f" The caption reads: '{caption}'."

    prompt += f" Question: {question}"

    if "unanswerable" in qa_type:
        return prompt.strip()
    else:
        if "closed-ended" in qa_type:
            if "finite answer set" in qa_type:
                if "binary" in qa_type:
                    prompt += " Please only answer 'yes' or 'no'."
                else:
                    # Normalize options: handle ABCD or A,B,C,D
                    if split in ("train", "validation") and "," not in answer_options and len(answer_options) == 4:
                        options = ",".join(list(answer_options))
                    else:
                        options = answer_options
                    prompt += f" Please choose from the following options: {options}."
        elif "infinite answer set" in qa_type:
            prompt += " Provide a short direct answer."
        elif "visual" in qa_type:
            prompt += " Pay attention to visual aspects like color, position, and shape."
        elif "non-visual" in qa_type:
            prompt += " Base your answer only on data values, not visual features."

    return prompt.strip()


def evaluate_model(processor, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_dataset = load_datasets(validation=True, train=False, test=False)
    total_rows = len(validation_dataset)

    results = []
    for i, entry in tqdm.tqdm(
        validation_dataset.iterrows(), total=total_rows, desc="Evaluating", unit="entry", unit_scale=True
    ):
        instance_id = entry["instance_id"]
        image_path = entry["image_file"]
        prompt_text = build_dynamic_prompt(entry, split="validation")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": load_real_image_path(image_path, validation=True)},
                    {"type": "text", "text": prompt_text},
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
        if device.type == "cuda":
            inputs = inputs.to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        answer = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        results.append({"instance_id": instance_id, "answer_pred": answer[0]})

    return results


def main():
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    results = evaluate_model(processor, model)
    # save results to a csv file
    prediction_file_path = path.join(PREDICITION_PATH, "predictions", "predictions.csv")
    if not path.exists(path.dirname(prediction_file_path)):
        print(f"Creating directory: {path.dirname(prediction_file_path)}")
        makedirs(path.dirname(prediction_file_path))

    # create df from results
    df = pd.DataFrame(results)
    df.to_csv(prediction_file_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"Predictions saved to {prediction_file_path}")


if __name__ == "__main__":
    main()
