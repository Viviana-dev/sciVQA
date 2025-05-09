import random
import sys
from os import environ, path
from pathlib import Path

import torch
import wandb
from peft import LoraConfig, TaskType, get_peft_model
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Qwen2VLProcessor,
)
from trl import SFTConfig, SFTTrainer

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from training.dataset import SciVQAConversationDataset
from training.gpu_cleaner import clear_memory

environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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
    target_modules: list[str],
) -> None:
    clear_memory()

    random.seed(42)
    torch.manual_seed(42)

    # Load the SciVQA conversation‑style datasets
    train_dataset: Dataset = SciVQAConversationDataset(split="train")
    eval_dataset: Dataset = SciVQAConversationDataset(split="validation")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=compute_dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_cache=False,
    )

    processor = Qwen2_5_VLProcessor.from_pretrained(model_name, use_fast=False)

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_rank,
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(model, peft_config)
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
        logging_steps=10,  # Steps interval for logging
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
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        # Hub and reporting
        report_to="wandb",  # Reporting tool for tracking metrics
        # Gradient checkpointing settings
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        # Dataset configuration
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        remove_unused_columns=False,  # Whether to remove unused columns in the dataset
    )

    wandb.init(
        project=f"qwen2.5-VL-7b-instruct-chart",
        name=f"{version}",
        config=training_args,
    )

    def collate_fn(batch):

        all_messages = [item["messages"] for item in batch]
        texts = processor.apply_chat_template(all_messages, tokenize=False)

        image_inputs, _ = process_vision_info(all_messages)

        batch = processor(
            text=texts,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        )

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
            image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
        else:
            image_tokens = [
                processor.tokenizer.convert_tokens_to_ids(processor.image_token)
            ]  # Convert image token to ID

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        batch["labels"] = labels

        return batch

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        processing_class=processor.tokenizer,
    )

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training complete.")
    # Save the model
    trainer.save_model(path.join(output_dir, "model"))


if __name__ == "__main__":
    trainLoraModel()
