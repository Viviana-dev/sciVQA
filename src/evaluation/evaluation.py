import csv
import shutil
import sys
from os import makedirs, path
from pathlib import Path
from typing import Literal
from unittest import result

import pandas as pd
import torch
import tqdm
from peft import PeftModel
from PIL import Image
from pytz import VERSION
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from helpers.build_prompt import build_dynamic_prompt
from helpers.constants import BASE_PATH, CSV_PATH, LORA_PATH, PREDICITION_PATH
from helpers.data_load import load_datasets, load_real_image_path
from helpers.qwen_util import custom_process_vision_info


def strip_cot(text: str) -> str:
    if "<answer>" in text:
        return text.split("<answer>", 1)[1].strip()
    return text.strip()


@torch.inference_mode()
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

    batch_size = 64
    processor.tokenizer.padding_side = "left"

    results = []
    for start in tqdm.tqdm(range(0, total_rows, batch_size), desc="Evaluating", unit="batch"):
        batch = dataset.iloc[start : start + batch_size]

        instance_ids = batch["instance_id"].tolist()
        image_paths = batch["image_file"].tolist()
        if dataset_type != "test":
            gold_answers = batch.get("answer", [""] * len(batch)).tolist()

        prompts_states = [
            build_dynamic_prompt(entry, split=dataset_type, save_sample_path=save_sample_path)
            for _, entry in batch.iterrows()
        ]
        prompt_texts = [p for p, _ in prompts_states]
        states = [s for _, s in prompts_states]

        messages_batch = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": load_real_image_path(img, **{dataset_type: True}),
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
            for img, prompt in zip(image_paths, prompt_texts)
        ]

        texts = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages_batch]
        images, _ = zip(*[custom_process_vision_info([m]) for m in messages_batch])

        inputs = processor(text=texts, images=list(images), padding=True, return_tensors="pt", padding_side="left").to(
            device
        )

        model.eval()
        eos_token_id = processor.tokenizer.eos_token_id
        pad_token_id = processor.tokenizer.pad_token_id
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        raw_outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        answers = [strip_cot(text) for text in raw_outputs]

        if dataset_type == "test":
            for instance_id, answer in zip(instance_ids, answers):
                results.append({"instance_id": instance_id, "answer_pred": answer})
        else:
            for instance_id, answer, gold, state in zip(instance_ids, answers, gold_answers, states):
                results.append({"instance_id": instance_id, "answer_pred": answer})
                save_path = Path(path.join(save_sample_path, instance_id))
                prompt_file_path = path.join(save_path, "prompt.txt")
                if state and path.exists(prompt_file_path):
                    if answer == gold:
                        shutil.rmtree(save_path)
                    else:
                        with open(prompt_file_path, "a", encoding="utf-8") as f:
                            f.write(f"\n\n\nAnswer: {answer}")
                            f.write(f"\nGold answer: {gold}")

    return results


def evaluate_model_predictions(
    adapter_path: str | None,
    model_name: str,
    version: str,
    dataset_type: Literal["train", "validation", "test"] = "validation",
):

    # load also the preprocessor form the adapter path
    if adapter_path is None:
        print("Start Zero Shot Evaluation...")
        processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    else:
        processor = AutoProcessor.from_pretrained(adapter_path, use_fast=False)
        base_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="sequential",
        )

        special_tokens_dict = {"additional_special_tokens": ["<box>", "</box>", "<thinking>", "<answer>"]}
        processor.tokenizer.add_special_tokens(special_tokens_dict)
        base_model.resize_token_embeddings(len(processor.tokenizer))
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


if __name__ == "__main__":
    VERSION = "Version_15"
    MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
    OUTPUT_DIR = Path(path.join(LORA_PATH, "no-ocr-v4", VERSION))
    MODEL_PATH = Path(path.join(OUTPUT_DIR, "model"))

    def save_df_back(split: str, df: pd.DataFrame):
        csv_path = path.join(CSV_PATH, f"{split}.csv")
        df.to_csv(csv_path, index=False)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=False)
    special_tokens_dict = {"additional_special_tokens": ["<box>", "</box>", "<thinking>", "<answer>"]}
    processor.tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(processor.tokenizer))
    model = PeftModel.from_pretrained(
        model,
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # result = list of {"instance_id": str, "answer_pred": str}
    results = evaluate_model(processor, model, Path(path.join(BASE_PATH, "sample", VERSION)), dataset_type="validation")

    # add the resuts to the dataset
    dataset = load_datasets(validation=True, train=False, test=False)
    dataset["answer_pred"] = ""
    for result in results:
        dataset.loc[dataset["instance_id"] == result["instance_id"], "answer_pred"] = result["answer_pred"]

    # save the dataset with the predictions
    save_df_back("validation", dataset)

    # same for train:
    dataset = load_datasets(validation=False, train=True, test=False)
    results = evaluate_model(processor, model, Path(path.join(BASE_PATH, "sample", VERSION)), dataset_type="train")
    dataset["answer_pred"] = ""
    for result in results:
        dataset.loc[dataset["instance_id"] == result["instance_id"], "answer_pred"] = result["answer_pred"]
    save_df_back("train", dataset)
