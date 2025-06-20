import csv
import logging
import sys
from os import makedirs, path
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from peft import PeftModel
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, Gemma3ForConditionalGeneration

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


from evaluation.scoring import compute_evaluation_scores
from helpers.constants import BASE_PATH, PREDICITION_PATH, SPECIAL_TOKENS
from helpers.dataset_utils import ChartQAEvaluationDataset, SciVQAEvaluationDataset
from helpers.logging_utils import setup_logger
from helpers.qwen_util import custom_process_vision_info

logger = setup_logger(__name__)


def strip_cot(text: str) -> str:
    """
    Strip chain-of-thought prefix up to <answer> tag.

    Parameters
    ----------
    text : str
        Generated text possibly containing a '<answer>' tag.

    Returns
    -------
    str
        The substring after '<answer>' tag, or the original text stripped.
    """
    if "<answer>" in text:
        return text.split("<answer>", 1)[1].strip()
    return text.strip()


def maybe_create_dir(dir_path: Path | str, print_log: bool = True):
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    dir_path : str
        Path to the directory to create.
    """
    if not path.exists(dir_path):
        if print_log:
            logger.info(f"Creating directory: {dir_path}")
        makedirs(dir_path)


def maybe_save_csv(df: pd.DataFrame, file_path: str):
    """
    Save a DataFrame to CSV and log the action.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save to CSV.
    file_path : str
        Destination path for the CSV file.
    """
    df.to_csv(file_path, index=False, quoting=csv.QUOTE_ALL)
    logger.info(f"Predictions saved to {file_path}")


@torch.inference_mode()
def evaluate_model(
    processor: AutoProcessor,
    model: AutoModelForImageTextToText | PeftModel,
    save_sample_path: Path,
    batches: DataLoader,
    dataset_type: Literal["train", "validation", "test"] = "validation",
    accelerator: Accelerator | None = None,
    dataset_len: int = 0,
    dataset_name: Literal["scivqa", "chartqa"] = "scivqa",
) -> list[dict[str, str]]:
    """
    Run model evaluation over batches and save mispredicted samples.

    Parameters
    ----------
    processor : AutoProcessor
        Processor for preparing inputs and decoding outputs.
    model : AutoModelForImageTextToText
        The image-text model used for generation.
    save_sample_path : Path
        Directory path to save sample prompts and images.
    batches : DataLoader
        DataLoader providing evaluation batches.
    dataset_type : Literal["train", "validation", "test"], default "validation"
        Split type being evaluated.
    accelerator : Accelerator or None, optional
        Accelerator instance for distributed evaluation.
    dataset_len : int, default 0
        Total number of samples for progress bar.
    dataset_name : Literal["scivqa", "chartqa"], default "scivqa"
        Name of the dataset being evaluated.

    Returns
    -------
    list[dict[str, str]]
        A list of prediction dicts containing 'instance_id' and 'answer_pred'.
    """
    processor.tokenizer.padding_side = "left"
    pbar = tqdm(total=dataset_len, desc="Evaluating", unit="Question", unit_scale=True)
    results = []
    for batch in batches:
        instance_ids: list[str] = batch["instance_id"]
        messages: list[list[dict[str, any]]] = batch["messages"]
        gold_answers: list[str] = batch["answer"]

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
        # cut at the end all `<end_of_turn>` tokens if model is Gemma3ForConditionalGeneration
        if isinstance(model, Gemma3ForConditionalGeneration):
            answers = [answer.split("<end_of_turn>")[0] for answer in answers]

        # Save wrong predicted samples with prompt and image
        for answer, instance_id, gold_answer, message in zip(answers, instance_ids, gold_answers, messages):
            results.append({"instance_id": instance_id, "answer_pred": answer})
            if dataset_type != "test":
                if answer != gold_answer:
                    save_path = Path(path.join(save_sample_path, instance_id))
                    maybe_create_dir(save_path, print_log=False)
                    prompt_file_path = path.join(save_path, "prompt.txt")
                    image_file_path = path.join(save_path, "image.png")
                    if dataset_name == "scivqa":
                        root_image_path = message[1]["content"][0]["image"]
                        image = Image.open(root_image_path).convert("RGB")
                        image.save(image_file_path)
                    elif dataset_name == "chartqa":
                        image = message[1]["content"][0]["image"].convert("RGB")
                        image.save(image_file_path)
                    with open(prompt_file_path, "w") as f:
                        f.write(f"Prompt: {message[1]['content'][1]['text']}\n")
                        f.write(f"Response: {answer}\n")
                        f.write(f"Gold Answer: {gold_answer}\n")

        if accelerator:
            accelerator.wait_for_everyone()
            pbar.update(accelerator.num_processes)
        else:
            pbar.update(len(batch))

    return results


def evaluate_model_predictions(
    adapter_path: str | Path | None,
    model_name: str,
    version: str,
    dataset_type: Literal["train", "validation", "test"] = "validation",
    accelerate: bool = False,
    scoring: bool = True,
    batch_size: int = 1,
    dataset_name: Literal["scivqa", "chartqa"] = "scivqa",
    dynamic_prompt: bool = False,
):
    """
    Load model and dataset, perform evaluation, and save or score predictions.

    Parameters
    ----------
    adapter_path : str or None
        Path to adapter for fine-tuning; if None, use zero-shot evaluation.
    model_name : str
        Pretrained model name or identifier.
    version : str
        Version identifier for saving predictions.
    dataset_type : Literal["train", "validation", "test"], default "validation"
        Split to evaluate.
    accelerate : Boolean , default False
        Whether to use an accelerator for distributed evaluation.
    scoring : bool, default True
        Whether to compute and log evaluation scores.
    batch_size : int, default 1
        Batch size for DataLoader.
    dataset_name : Literal["scivqa", "chartqa"], default "scivqa"
        Name of the dataset to evaluate.
    dynamic_prompt : bool, default False
        Whether to use dynamic prompts for evaluation, or the more generalized one which fits all datasets.
    """
    accelerator = Accelerator() if accelerate else None

    global logger
    logger = setup_logger(__name__, level=logging.INFO, accelerator=accelerator)

    if dataset_name == "scivqa":
        dataset = SciVQAEvaluationDataset(split=dataset_type, dynamic=dynamic_prompt)
    elif dataset_name == "chartqa":
        dataset = ChartQAEvaluationDataset(split=dataset_type)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, pin_memory=True)
    device_map = None if accelerator else "auto"

    if adapter_path is None:
        logger.info("Start Zero Shot Evaluation...")
        processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager" if "gemma" in model_name else "flash_attention_2",
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

        processor.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        base_model.resize_token_embeddings(len(processor.tokenizer))

        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )

    if accelerator:
        model, dataloader = accelerator.prepare(model, dataloader)
        accelerator.wait_for_everyone()

    sample_path = Path(path.join(BASE_PATH, "sample", version))

    results = evaluate_model(
        processor, model, sample_path, dataloader, dataset_type, accelerator, len(dataset) // batch_size, dataset_name
    )

    # Gather results if using accelerator
    if accelerator:
        accelerator.wait_for_everyone()
        all_results = gather_object(results)
    else:
        all_results = results

    # Save predictions and optionally compute scores on the main process or when not using accelerator
    if not accelerator or accelerator.is_main_process:
        prediction_dir = path.dirname(path.join(PREDICITION_PATH, "predictions", "predictions.csv"))
        maybe_create_dir(prediction_dir)
        df = pd.DataFrame(all_results)
        maybe_save_csv(df, path.join(PREDICITION_PATH, "predictions", "predictions.csv"))
        if scoring and dataset_type != "test":
            compute_evaluation_scores(version=version, dataset_name=dataset_name)
        if accelerator:
            accelerator.end_training()
