import random
import shutil
import sys
from os import environ, path
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, broadcast
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen2VLProcessor
from trl import SFTConfig, SFTTrainer

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from helpers.constants import LOARA_VERSIONS_PATH, SPECIAL_TOKENS
from helpers.dataset_utils import SciVQATrainingDataset
from helpers.logging_utils import setup_logger
from helpers.qwen_util import custom_process_vision_info

environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
environ.setdefault("WANDB_SILENT", "true")
environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
environ.setdefault("TORCH_USE_CUDA_DSA", "1")

logger = setup_logger(__name__)


def trainLoraModel(
    model_name: str,
    version: str,
    output_dir: Path,
    batch_size: int,
    grad_acc: int,
    epochs: int,
    lr: float,
    dtype: torch.dtype = torch.bfloat16,
    lora_rank: int = 64,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | str = "all-linear",
    accelerate: bool = False,
):
    """
    Train a LoRA model using the specified parameters.

    Parameters
    ----------
    model_name : str
        The name of the model to be trained.
    version : str
        The version of the model.
    output_dir : Path
        The directory where the model will be saved.
    batch_size : int
        The batch size for training.
    grad_acc : int
        The number of gradient accumulation steps.
    epochs : int
        The number of epochs for training.
    lr : float
        The learning rate for training.
    dtype : torch.dtype, default torch.bfloat16
        The data type for training (e.g., torch.float16, torch.bfloat16).
    lora_rank : int, default 64
        The rank of the LoRA model.
    lora_alpha : int, default 32
        The alpha value for the LoRA model.
    lora_dropout : float, default 0.05
        The dropout rate for the LoRA model.
    target_modules : list[str] | str, default "all-linear"
        The target modules for the LoRA model.
    accelerate : Boolean, default False
        Whether to use the accelerator for distributed training.
    """
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = None if not accelerate else Accelerator(kwargs_handlers=[kwargs], log_with="wandb")
    model_dir = Path(path.join(output_dir, "model"))
    if accelerator:
        logger = setup_logger(__name__, accelerator=accelerator)

    device_map = None if accelerator else "auto"
    device = accelerator.device if accelerator else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)
    torch.manual_seed(42)

    # Load the SciVQA conversation‑style datasets
    train_dataset: Dataset = SciVQATrainingDataset(split="train")
    eval_dataset: Dataset = SciVQATrainingDataset(split="validation")

    # check if model exist or if it is empty
    accelerator.wait_for_everyone()

    if model_dir.exists() and any(model_dir.iterdir()):
        if accelerator.is_main_process:
            ans = input(f"Model already exists at {model_dir}. Delete it? (y/N): ")
            delete_flag = ans.strip().lower() == "y"
        else:
            delete_flag = False

        # broadcast the decision
        delete_tensor = torch.tensor(int(delete_flag), device=accelerator.device)
        broadcast(delete_tensor)

        # main rank deletes if requested
        if delete_tensor.item() == 1 and accelerator.is_main_process:
            shutil.rmtree(model_dir)
            logger.info(f"Deleted existing model in {model_dir}.")

        accelerator.wait_for_everyone()

        if delete_tensor.item() == 0:
            logger.info("Skip Training!!")
            return

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_cache=False,
    )
    if accelerator:
        model = model.to(device=device, dtype=dtype)

    processor: AutoProcessor = AutoProcessor.from_pretrained(model_name, use_fast=False)

    processor.tokenizer.add_special_tokens(SPECIAL_TOKENS)
    processor.tokenizer.padding_side = "left"

    model.resize_token_embeddings(len(processor.tokenizer))

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_rank,
        bias="none",
        target_modules=target_modules,
        modules_to_save=["lm_head", "embed_tokens"],
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, peft_config)

    if accelerator:
        if accelerator.is_main_process:
            peft_model.print_trainable_parameters()
    else:
        peft_model.print_trainable_parameters()

    if accelerator:
        accelerator.wait_for_everyone()

    training_args = SFTConfig(
        run_name=f"{version}",
        output_dir=output_dir,  # Directory to save the model
        num_train_epochs=epochs,  # Number of training epochs
        per_device_train_batch_size=batch_size,  # Batch size for training
        per_device_eval_batch_size=batch_size,  # Batch size for evaluation
        gradient_accumulation_steps=grad_acc,  # Steps to accumulate gradients
        gradient_checkpointing=True,
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=lr,  # Learning rate for training
        lr_scheduler_type="cosine",  # Type of learning rate scheduler
        # Logging and evaluation
        logging_steps=100,  # Steps interval for logging
        eval_steps=200,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=200,  # Steps interval for saving
        metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        greater_is_better=False,  # Whether higher metric values are better
        save_total_limit=3,  # Limit the number of saved models
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        max_grad_norm=1.0,  # gradient clipping for stability
        max_length=512,
        warmup_ratio=0.05,  # Ratio of total steps for warmup -> 5%
        # Hub and reporting
        report_to="wandb",  # Reporting tool for tracking metrics
        # Gradient checkpointing settings
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        # Dataset configuration
        dataloader_num_workers=4,
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        remove_unused_columns=False,  # Whether to remove unused columns in the dataset
        label_names=["labels"],
        use_liger_kernel=True,
        ddp_find_unused_parameters=False,
    )

    if accelerator:
        accelerator.init_trackers(
            project_name=f"{model_name.split('/')[-1]}-chart",
            config=training_args,
            init_kwargs={
                "wandb": {
                    "name": f"{version}",
                }
            },
        )
    else:
        wandb.init(
            project=f"{model_name.split('/')[-1]}-chart",
            name=f"{version}",
            config=training_args,
        )

    def collate_fn(batch):
        """Collate function to process a batch of data.

        Parameters
        ----------
        batch : list[dict]
            A list of dictionaries containing the data for each sample in the batch.

        Returns
        -------
        dict
            A dictionary containing the collated data for the batch.
        """
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
        for k, v in batch.items():
            if torch.is_floating_point(v) and v.dtype != dtype:
                batch[k] = v.to(dtype=dtype)
        batch["attention_mask"] = batch["attention_mask"].to(torch.bool)

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        pad_id = processor.tokenizer.pad_token_id

        image_ids = (
            [151652, 151653, 151655]  # Qwen‑2 VL internal image tokens
            if isinstance(processor, Qwen2VLProcessor)
            else [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
        )

        static_mask_ids = torch.tensor([pad_id, *image_ids], device=labels.device)
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
    )

    if accelerator:
        trainer = accelerator.prepare(trainer)

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")

    trainer.save_model(model_dir)
    if accelerator:
        unwrapped_model = accelerator.unwrap_model(peft_model)
        unwrapped_model.save_pretrained(model_dir, save_embedding_layers=True)
    else:
        # Save the model
        peft_model.save_pretrained(model_dir, save_embedding_layers=True)
    # Save the processor
    processor.save_pretrained(model_dir, save_embedding_layers=True)

    if not accelerator or accelerator.is_main_process:
        lora_versions_path = Path(path.join(LOARA_VERSIONS_PATH, version))
        if not lora_versions_path.exists():
            lora_versions_path.mkdir(parents=True, exist_ok=True)
        # move adapter_config.json to lora_versions_path
        config_path = Path(path.join(output_dir, "model", "adapter_config.json"))
        if path.exists(config_path):
            logger.info(f"Copy adapter_config.json to {lora_versions_path}")
            shutil.copyfile(config_path, path.join(lora_versions_path, "adapter_config.json"))
        else:
            logger.info(f"adapter_config.json not found in {model_dir}")

        # clean checkpoints except the folder named 'model'
        for item in output_dir.iterdir():
            if item.is_dir() and item.name != "model":
                shutil.rmtree(item)
                logger.info(f"Removed {item}")
            else:
                logger.info(f"Skipped {item}")
    if accelerator:
        accelerator.wait_for_everyone()
