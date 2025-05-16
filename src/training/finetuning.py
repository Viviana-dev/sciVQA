import copy
import random
import shutil
import sys
from os import environ, path
from pathlib import Path

import torch
import wandb
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from rouge_score import rouge_scorer
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, EarlyStoppingCallback, Qwen2VLProcessor
from trl import SFTConfig, SFTTrainer

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


from helpers.constants import LOARA_VERSIONS_PATH
from helpers.qwen_util import custom_process_vision_info
from training.dataset import SciVQAConversationDataset

environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
environ.setdefault("WANDB_SILENT", "true")
environ["CUDA_LAUNCH_BLOCKING"] = "1"
environ["TORCH_USE_CUDA_DSA"] = "1"

rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def trainLoraModel(
    model_name: str,
    version: str,
    output_dir: Path,
    batch_size: int,
    grad_acc: int,
    epochs: int,
    lr: float,
    compute_dtype: torch.dtype,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str] | str,
) -> None:
    model_dir = Path(path.join(output_dir, "model"))

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Load the SciVQA conversation‑style datasets
    train_dataset: Dataset = SciVQAConversationDataset(split="train")
    eval_dataset: Dataset = SciVQAConversationDataset(split="validation")

    # train_chartqa_dataset: Dataset = ChartQAConversationDataset(split="train")
    # train_dataset = ConcatDataset([train_dataset, train_chartqa_dataset])

    # check if model exist or if it is empty
    if path.exists(model_dir) and len(list(model_dir.iterdir())) > 0:
        print(f"Model already exists in {model_dir}.")

        model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            torch_dtype=compute_dtype,
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_cache=False,
        )

        processor = (AutoProcessor.from_pretrained(model_dir, use_fast=False),)
        special_tokens_dict = {"additional_special_tokens": ["<box>", "</box>", "<thinking>", "<answer>"]}
        processor.tokenizer.add_special_tokens(special_tokens_dict)
        processor.tokenizer.padding_side = "left"

        model.resize_token_embeddings(len(processor.tokenizer))

        peft_model = PeftModel.from_pretrained(
            model,
            model_dir,
            torch_dtype=compute_dtype,
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_cache=False,
            is_trainable=True,
        )

        peft_model.resize_token_embeddings(len(processor.tokenizer))
    else:
        print(f"Model does not exist in {model_dir}.")
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=compute_dtype,
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_cache=False,
        )

        processor = AutoProcessor.from_pretrained(model_name, use_fast=False)

        special_tokens_dict = {"additional_special_tokens": ["<box>", "</box>", "<thinking>", "<answer>"]}
        processor.tokenizer.add_special_tokens(special_tokens_dict)
        processor.tokenizer.padding_side = "left"

        model.resize_token_embeddings(len(processor.tokenizer))

        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_rank,
            bias="none",
            target_modules=target_modules,
            modules_to_save=["lm_head", "model.embed_tokens"],
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(model, peft_config)

        peft_model.resize_token_embeddings(len(processor.tokenizer))
        peft_model.print_trainable_parameters()

    training_args = SFTConfig(
        run_name=f"LoRa-{version}",
        output_dir=output_dir,  # Directory to save the model
        num_train_epochs=epochs,  # Number of training epochs
        per_device_train_batch_size=batch_size,  # Batch size for training
        per_device_eval_batch_size=batch_size,  # Batch size for evaluation
        gradient_accumulation_steps=grad_acc,  # Steps to accumulate gradients
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=lr,  # Learning rate for training
        lr_scheduler_type="cosine",  # Type of learning rate scheduler
        # Logging and evaluation
        logging_steps=100,  # Steps interval for logging
        eval_steps=100,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=100,  # Steps interval for saving
        metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        greater_is_better=False,  # Whether higher metric values are better
        load_best_model_at_end=True,  # Load the best model after training
        save_total_limit=3,  # Limit the number of saved models
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        max_grad_norm=1.0,  # gradient clipping for stability
        max_length=128,
        warmup_ratio=0.05,  # Ratio of total steps for warmup -> 5%
        # Hub and reporting
        report_to="wandb",  # Reporting tool for tracking metrics
        # Gradient checkpointing settings
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        # Dataset configuration
        dataloader_num_workers=16,
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        remove_unused_columns=False,  # Whether to remove unused columns in the dataset
        label_names=["labels"],
    )

    wandb.init(
        project=f"{model_name.split('/')[-1]}-chart",
        name=f"{version}",
        config=training_args,
    )

    early_stop_cb = EarlyStoppingCallback(
        early_stopping_patience=4,  # how many eval rounds to wait
        early_stopping_threshold=0.0001,  # require a strictly better score
    )

    def collate_fn(batch):
        texts, images = [], []
        for item in batch:
            image_inputs, _ = custom_process_vision_info(item["messages"])
            txt = processor.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=False,
            ).strip()
            texts.append(txt)
            images.append(image_inputs)

        batch = processor(text=texts, images=images, padding=True, return_tensors="pt")
        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        pad_id = processor.tokenizer.pad_token_id
        box_ids = [
            processor.tokenizer.convert_tokens_to_ids("<box>"),
            processor.tokenizer.convert_tokens_to_ids("</box>"),
        ]
        image_ids = (
            [151652, 151653, 151655]  # Qwen‑2 VL internal image tokens
            if isinstance(processor, Qwen2VLProcessor)
            else [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
        )

        static_mask_ids = torch.tensor([pad_id, *box_ids, *image_ids], device=labels.device)
        labels[torch.isin(labels, static_mask_ids)] = -100

        answer_id = processor.tokenizer.convert_tokens_to_ids("<answer>")
        answer_positions = (input_ids == answer_id).nonzero(as_tuple=False)
        for batch_idx, pos in answer_positions:
            labels[batch_idx, : pos + 1] = -100  # mask CoT + <answer> token

        batch["labels"] = labels
        return batch

    trainer: SFTTrainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        processing_class=processor.tokenizer,
        callbacks=[early_stop_cb],
    )

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training complete.")
    # Save the model
    trainer.save_model(model_dir)

    # save processor
    processor.save_pretrained(model_dir)

    lora_versions_path = Path(path.join(LOARA_VERSIONS_PATH, version))
    if not lora_versions_path.exists():
        lora_versions_path.mkdir(parents=True, exist_ok=True)
    # move adapter_config.json to lora_versions_path
    config_path = Path(path.join(output_dir, "model", "adapter_config.json"))
    if path.exists(config_path):
        print(f"Copy adapter_config.json to {lora_versions_path}")
        shutil.copyfile(
            config_path,
            path.join(lora_versions_path, "adapter_config.json"),
        )
    else:
        print(f"adapter_config.json not found in {model_dir}")

    # clean checkpoints except the folder named 'model'
    for item in output_dir.iterdir():
        if item.is_dir() and item.name != "model":
            shutil.rmtree(item)
            print(f"Removed {item}")
        else:
            print(f"Skipped {item}")
