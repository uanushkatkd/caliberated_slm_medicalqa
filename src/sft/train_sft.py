# src/sft/train_sft.py

import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from src.models.load_model import load_base_model, get_device
from src.models.lora_config import get_lora_config
from peft import get_peft_model


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
TRAIN_FILE = "data/processed/sft_train.jsonl"
VAL_FILE = "data/processed/sft_val.jsonl"
OUTPUT_DIR = "outputs/sft"


def load_sft_dataset():
    return load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "validation": VAL_FILE
        }
    )


MAX_LENGTH = 512  # Reduced for 16GB RAM


def tokenize_fn(example, tokenizer):
    text = example["prompt"] + "\n" + example["response"]
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main():
    device = get_device()
    print(f"Using device: {device}")

    # Load model in fp16 to save memory (~6GB instead of ~12GB)
    model, tokenizer = load_base_model(MODEL_NAME, device=device, use_fp16=True)

    # Enable gradient checkpointing BEFORE wrapping with PEFT
    model.gradient_checkpointing_enable()

    lora_config = get_lora_config(r=4, lora_alpha=8)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_sft_dataset()

    tokenized = dataset.map(
        lambda x: tokenize_fn(x, tokenizer),
        remove_columns=dataset["train"].column_names,
        batched=False
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # MPS doesn't support fp16 mixed-precision training
    use_fp16 = device == "cuda"

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=use_fp16,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        dataloader_pin_memory=False,
        gradient_checkpointing=True,  # Saves memory by recomputing activations
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
