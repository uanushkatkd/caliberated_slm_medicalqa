# src/sft/train_sft.py

import json
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from src.models.load_model import load_base_model
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


def tokenize_fn(example, tokenizer):
    text = example["prompt"] + "\n" + example["response"]
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding=False
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main():
    model, tokenizer = load_base_model(MODEL_NAME)
    lora_config = get_lora_config(r=4, lora_alpha=8)
    model = get_peft_model(model, lora_config)

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

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        report_to="none"
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
