import random
import sys
from os import environ, makedirs, path
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from helpers.constants import LORA_PATH
from training.dataset import SciVQAConversationDataset

local_rank = None

# Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in the environment
# to allow for dynamic memory allocation

environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def rank0_print(*args):
    if local_rank == 0 or local_rank == "0" or local_rank is None:
        print(*args)


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def main() -> None:
    global local_rank

    SEED = 42
    MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
    VERSION = "no-ocr-v4"
    OUTPUT_DIR = Path(path.join(LORA_PATH, VERSION))
    if not OUTPUT_DIR.exists():
        makedirs(OUTPUT_DIR, exist_ok=True)
    BATCH_SIZE = 12
    GRAD_ACC = 2
    EPOCHS = 25
    LR = 1e-4
    COMPUTE_DTYPE = torch.bfloat16
    LORA_RANK = 32  # Lora rank is the number of low-rank matrices to be used in the LoRA module: higher rank means more parameters
    LORA_ALPHA = 32  # Lora alpha is the scaling factor for the low-rank matrices: higher alpha means more parameters
    LORA_DROPOUT = 0.05
    FREEZE_VISION = True
    FREEZE_MERGER = True
    FREEZE_LLM = False

    random.seed(SEED)
    torch.manual_seed(SEED)

    # Load the SciVQA conversation‑style datasets
    train_dataset = SciVQAConversationDataset(split="train")
    eval_dataset = SciVQAConversationDataset(split="validation")

    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    processor.padding_side = "left"

    def collate_fn(batch):

        all_messages = [item["messages"] for item in batch]

        texts = processor.apply_chat_template(all_messages, tokenize=False, add_generation_prompt=False)

        image_inputs, video_inputs = process_vision_info(all_messages)

        model_inputs = processor(
            padding_side="left",
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # labels are just the input_ids, shifted inside `model.prepare_decoder_input_ids_from_labels`
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs

    # Load the model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=COMPUTE_DTYPE,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    # freeze vision and merger parameters
    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not FREEZE_VISION)
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not FREEZE_MERGER)

    # unfreeze llm
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not FREEZE_LLM)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not FREEZE_LLM)

    lora_namespan_exclude = ["lm_head", "embed_tokens"]

    if not FREEZE_LLM:
        lora_namespan_exclude += ["visual"]

    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=find_target_linear_names(
            model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=-1
        ),
        lora_dropout=LORA_DROPOUT,
    )

    rank0_print("Adding LoRA to the model...")
    model = get_peft_model(model, peft_config)

    if not FREEZE_VISION:
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad = True

    if not FREEZE_MERGER:
        for name, param in model.named_parameters():
            if "merger" in name:
                param.requires_grad = True

    # Set up the training arguments
    training_args = TrainingArguments(
        run_name=f"Lora-{VERSION}",
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_dir=path.join(OUTPUT_DIR, "logs"),
        logging_steps=100,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_32bit",
        report_to="wandb",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        label_names=["labels"],
    )

    data_collator = collate_fn
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )

    # Start training
    rank0_print("Starting training...")
    trainer.train()
    rank0_print("Training complete.")
    # Save the model
    rank0_print("Saving the model...")
    trainer.save_model(OUTPUT_DIR)
    rank0_print("Model saved.")
    rank0_print("Saving the processor...")
    processor.save_pretrained(OUTPUT_DIR)
    rank0_print("Processor saved.")
    rank0_print("Saving the training arguments...")
    training_args.save_to_json(path.join(OUTPUT_DIR, "training_args.json"))
    rank0_print("Training arguments saved.")
    rank0_print("Saving the LoRA config...")
    model.peft_config.save_pretrained(OUTPUT_DIR)
    rank0_print("LoRA config saved.")
    rank0_print("Saving the model config...")
    model.config.save_pretrained(OUTPUT_DIR)
    rank0_print("Model config saved.")


if __name__ == "__main__":
    main()
