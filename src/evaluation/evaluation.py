import csv
import shutil
import sys
from asyncio import gather
from os import makedirs, path
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from dataset import SciVQAConversationDataset
from scoring import compute_evaluation_scores

from helpers.constants import BASE_PATH, LORA_PATH, PREDICITION_PATH
from helpers.qwen_util import custom_process_vision_info


def strip_cot(text: str) -> str:
    if "<answer>" in text:
        return text.split("<answer>", 1)[1].strip()
    return text.strip()


def _maybe_create_dir(dir_path: str):
    if not path.exists(dir_path):
        print(f"Creating directory: {dir_path}")
        makedirs(dir_path)


def _maybe_save_csv(df: pd.DataFrame, file_path: str):
    df.to_csv(file_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"Predictions saved to {file_path}")


@torch.inference_mode()
def evaluate_model(
    processor: AutoProcessor,
    model: AutoModelForImageTextToText,
    save_sample_path: Path,
    batches: DataLoader,
    dataset_type: Literal["train", "validation", "test"] = "validation",
    accelerator: Accelerator | None = None,
    dataset_len: int = 0,
):
    processor.tokenizer.padding_side = "left"
    pbar = tqdm(total=dataset_len, desc="Evaluating", unit="Question", unit_scale=True)
    results = []
    for batch in batches:
        instance_ids = batch["instance_id"]
        messages = batch["messages"]
        gold_answers = batch["answer"]

        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        images, _ = custom_process_vision_info(messages=messages)

        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt", padding_side="left").to(
            accelerator.device if accelerator else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        eos_token_id = processor.tokenizer.eos_token_id
        pad_token_id = processor.tokenizer.pad_token_id
        generate_fn = model.module.generate if hasattr(model, "module") else model.generate
        generated_ids = generate_fn(
            **inputs,
            max_new_tokens=128,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        raw_outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        answers = [strip_cot(output) for output in raw_outputs]

        for answer, instance_id, gold_answer in zip(answers, instance_ids, gold_answers):
            results.append({"instance_id": instance_id, "answer_pred": answer})
            if dataset_type != "test":
                save_path = Path(path.join(save_sample_path, instance_id))
                prompt_file_path = path.join(save_path, "prompt.txt")
                if path.exists(prompt_file_path):
                    if answer == gold_answer:
                        shutil.rmtree(save_path)
                    else:
                        with open(prompt_file_path, "a", encoding="utf-8") as f:
                            f.write(f"\n\n\nAnswer: {answer}")
                            f.write(f"\nGold answer: {gold_answer}")

        if accelerator:
            accelerator.wait_for_everyone()
            pbar.update(accelerator.num_processes)
        else:
            pbar.update(len(batch))

    return results


def evaluate_model_predictions(
    adapter_path: str | None,
    model_name: str,
    version: str,
    dataset_type: Literal["train", "validation", "test"] = "validation",
    accelerate: bool = False,
    scoring: bool = True,
    batch_size: int = 1,
):
    accelerator = Accelerator() if accelerate else None
    if accelerator and accelerator.is_main_process:
        print("Accelerator is initialized.")
    elif accelerator is None:
        print("Accelerator is not initialized.")

    dataset = SciVQAConversationDataset(split=dataset_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, pin_memory=True)
    device_map = {"": accelerator.process_index} if accelerate else "auto"

    if adapter_path is None:
        print("Start Zero Shot Evaluation...")
        processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_map,
        )
    else:
        processor = AutoProcessor.from_pretrained(adapter_path, use_fast=False)
        base_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_map,
        )

        special_tokens_dict = {"additional_special_tokens": ["<box>", "</box>", "<thinking>", "<answer>"]}
        processor.tokenizer.add_special_tokens(special_tokens_dict)
        base_model.resize_token_embeddings(len(processor.tokenizer))

        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )

    if accelerator:
        model, dataloader = accelerator.prepare(model, dataloader)

    # model.eval()
    sample_path = Path(path.join(BASE_PATH, "sample", version))

    if accelerate:
        accelerator.wait_for_everyone()

    results = evaluate_model(
        processor, model, sample_path, dataloader, dataset_type, accelerator, len(dataset) // batch_size
    )

    if accelerate:
        accelerator.wait_for_everyone()
        gathered_results = gather_object(results) if accelerator else results

    if accelerate and accelerator.is_main_process:
        print("Gathered results len:", len(gathered_results))
        prediction_dir = path.dirname(path.join(PREDICITION_PATH, "predictions", "predictions.csv"))
        _maybe_create_dir(prediction_dir)

        df = pd.DataFrame(gathered_results)
        _maybe_save_csv(df, path.join(PREDICITION_PATH, "predictions", "predictions.csv"))

        if scoring:
            if TYPE != "test":
                compute_evaluation_scores(version=VERSION)
    else:
        print("Results len:", len(results))
        prediction_dir = path.dirname(path.join(PREDICITION_PATH, "predictions", "predictions.csv"))
        _maybe_create_dir(prediction_dir)
        df = pd.DataFrame(results)
        _maybe_save_csv(df, path.join(PREDICITION_PATH, "predictions", "predictions.csv"))
        if scoring:
            if TYPE != "test":
                compute_evaluation_scores(version=VERSION)


if __name__ == "__main__":
    VERSION = "Version_21"
    TYPE = "validation"  # "train", "validation", "test"
    ADAPTER_PATH = Path(path.join(LORA_PATH, "no-ocr-v4", VERSION, "model"))
    MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
    evaluate_model_predictions(
        adapter_path=ADAPTER_PATH,
        model_name=MODEL_NAME,
        version=VERSION,
        dataset_type=TYPE,  # "train", "validation", "test"
        accelerate=True,
        scoring=True,
        batch_size=1,
    )
