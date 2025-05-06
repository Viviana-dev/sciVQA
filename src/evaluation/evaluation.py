import ast
import csv
import random
import re
import sys
from os import makedirs, path
from pathlib import Path

import pandas as pd
import torch
import tqdm
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from helpers.constants import BASE_PATH, PREDICITION_PATH
from helpers.data_load import load_datasets, load_real_image_path
from preprocessing.graph_prep import process_single_data_row
from preprocessing.ocr_utils import visualize_ocr_boxes


def parse_qa_types(qa_type_raw: str) -> set[str]:
    qa_str = str(qa_type_raw).lower()

    ordered_tokens = [
        "closed-ended",
        "unanswerable",
        "infinite answer set",
        "finite answer set",
        "non-binary",
        "binary",
        "non-visual",
        "visual",
    ]

    found = set()
    for token in ordered_tokens:
        # match token as a whole word; allow spaces or semicolons as separators
        pattern = r"(?:^|[\s;])" + re.escape(token) + r"(?:[\s;]|$)"
        if re.search(pattern, qa_str):
            found.add(token)
            # strip out the matched portion to prevent nested matches
            qa_str = re.sub(pattern, " ", qa_str, count=1)

    return found


def build_dynamic_prompt(entry, split="validation"):
    instance_id = entry["instance_id"]
    image_path = entry["image_file"]
    question = entry["question"]
    qa_type_raw = entry["qa_pair_type"]
    caption = entry.get("caption", "")
    figure_type = entry.get("figure_type", "figure")
    compound = entry.get("compound", False)
    figs_numb = entry.get("figs_numb", 0)
    answer_options = entry.get("answer_options", "")
    random_state: bool = random.choice([True, False])

    qa_types = parse_qa_types(qa_type_raw)

    # ── Figure metadata ───────────────────────────────────────────────────────
    prompt = f"You are looking at a {figure_type}"
    if compound:
        prompt += f" with {figs_numb} subfigures"
    prompt += "."

    if caption:
        prompt += f"\nThe caption reads: '{caption}'."

    # ── OCR region descriptions ───────────────────────────────────────────────
    ocr_boxes_norm, boxes = process_single_data_row(entry, split_name=split)
    region_lines = []
    for x1, y1, x2, y2, text in ocr_boxes_norm:
        x1, y1, x2, y2 = map(lambda v: round(v, 2), (x1, y1, x2, y2))
        region_lines.append(f'<box>({x1},{y1},{x2},{y2}): "{text.strip()}"</box>')
    region_block = "\n".join(region_lines)
    prompt += "\nThe following chart regions were detected with OCR (they may contain errors):\n" + region_block + "\n"
    prompt += f"\nQuestion: {question}"

    # ── Early exit for unanswerable cases ─────────────────────────────────────
    if "unanswerable" in qa_types:
        prompt += (
            "\nIf the answer cannot be inferred from the figure and caption, reply that the answer is not answerable !!"
            "\nResponse:"
        )
        return prompt.strip(), random_state

    # ── Visual / non‑visual cues ──────────────────────────────────────────────
    if "visual" in qa_types:
        prompt += "\n[Visual cue] Pay attention to colour, position, shape, size," " height, or direction."
    elif "non-visual" in qa_types:
        prompt += "\n[Data-only cue] Base your answer on numeric or textual values," " not visual features."

    # ── Answer‑set guidance ───────────────────────────────────────────────────
    if "infinite answer set" in qa_types:
        prompt += "\nRespond with a concise, one-word or very short phrase. No full sentences, no explanations."
    elif "finite answer set" in qa_types:
        if "binary" in qa_types:
            prompt += "\nPlease answer with 'Yes' or 'No' only."
        else:
            parsed_options = ast.literal_eval(answer_options)
            options = {k: v for d in parsed_options for k, v in d.items()}
            prompt += f"\nPlease choose one of the following options: {options}."
            prompt += "\nRespond only with the choosen options keyword, no explanations and no full sentences."

    # ── Final question & fallback ─────────────────────────────────────────────
    prompt += "\nResponse:"

    # choose randomly betwwen true and false

    if random_state:
        save_path = Path(path.join(BASE_PATH, "sample", instance_id))
        save_path.mkdir(parents=True, exist_ok=True)
        prompt_file_path = path.join(save_path, "prompt.txt")
        image_file_path = path.join(save_path, "image.png")
        root_image_path = load_real_image_path(image_path, **{split: True})
        image = Image.open(root_image_path).convert("RGB")
        visualize_ocr_boxes(image, boxes=boxes, color="red", show=False, save_path=image_file_path)
        with open(prompt_file_path, "w", encoding="utf-8") as f:
            f.write(f"QA-Type: {qa_type_raw}\n\n{prompt}")

    return prompt.strip(), random_state


def evaluate_model(processor, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_dataset = load_datasets(validation=True, train=False, test=False)
    total_rows = len(validation_dataset)

    results = []
    for _, entry in tqdm.tqdm(
        validation_dataset.iterrows(), total=total_rows, desc="Evaluating", unit="entry", unit_scale=True
    ):
        instance_id = entry["instance_id"]
        image_path = entry["image_file"]
        gold_answer = entry.get("answer", "")
        prompt_text, state = build_dynamic_prompt(entry, split="validation")

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
        generated_ids = model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        answer = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        results.append({"instance_id": instance_id, "answer_pred": answer[0]})

        # save the answer to the prompt.txt if the file exists
        save_path = Path(path.join(BASE_PATH, "sample", instance_id))
        prompt_file_path = path.join(save_path, "prompt.txt")
        if state and path.exists(prompt_file_path):
            with open(prompt_file_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n\nAnswer: {answer[0]}")
                f.write(f"\nGold answer: {gold_answer}")

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
