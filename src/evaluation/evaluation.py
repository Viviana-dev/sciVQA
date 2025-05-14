import csv
import shutil
import sys
from os import makedirs, path
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
import tqdm
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from helpers.build_prompt import build_dynamic_prompt
from helpers.constants import BASE_PATH, PREDICITION_PATH
from helpers.data_load import load_datasets, load_real_image_path
from helpers.qwen_util import custom_process_vision_info


def evaluate_model(
    processor, model, save_sample_path: Path, dataset_type: Literal["train", "validation", "test"] = "validation"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset_type == "train":
        dataset = load_datasets(validation=False, train=True, test=False)
    elif dataset_type == "test":
        dataset = load_datasets(validation=False, train=False, test=True)
    else:
        # default to validation
        dataset = load_datasets(validation=True, train=False, test=False)
    total_rows = len(dataset)

    results = []
    for _, entry in tqdm.tqdm(dataset.iterrows(), total=total_rows, desc="Evaluating", unit="entry", unit_scale=True):
        instance_id = entry["instance_id"]
        image_path = entry["image_file"]
        gold_answer = entry.get("answer", "")
        prompt_text, state = build_dynamic_prompt(entry, split=dataset_type)
        system_message: str = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_message,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": load_real_image_path(image_path, **{dataset_type: True}),
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            },
        ]

        text: str = processor.apply_chat_template(
            messages[1:2], tokenize=False, add_generation_prompt=True  # Use the message without the system message
        )
        image_inputs, video_inputs = custom_process_vision_info(messages)

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
        save_path = Path(path.join(save_sample_path, instance_id))
        prompt_file_path = path.join(save_path, "prompt.txt")
        if state and path.exists(prompt_file_path):
            if answer[0] == gold_answer:
                # remove the folder even if the folder is not empty
                shutil.rmtree(save_path)
            else:
                with open(prompt_file_path, "a", encoding="utf-8") as f:
                    f.write(f"\n\n\nAnswer: {answer[0]}")
                    f.write(f"\nGold answer: {gold_answer}")

    return results


def evaluate_model_predictions(
    adapter_path: str,
    model_name: str,
    version: str,
    dataset_type: Literal["train", "validation", "test"] = "validation",
):
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    sample_path = Path(path.join(BASE_PATH, "sample", version))

    results = evaluate_model(processor, model, sample_path, dataset_type)
    # save results to a csv file
    prediction_file_path = path.join(PREDICITION_PATH, "predictions", "predictions.csv")
    if not path.exists(path.dirname(prediction_file_path)):
        print(f"Creating directory: {path.dirname(prediction_file_path)}")
        makedirs(path.dirname(prediction_file_path))

    # create df from results
    df = pd.DataFrame(results)
    df.to_csv(prediction_file_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"Predictions saved to {prediction_file_path}")
