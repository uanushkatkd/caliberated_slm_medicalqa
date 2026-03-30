# src/sft/train_sft.py

import argparse
import torch
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    
)

from src.models.load_model import load_base_model, get_device
from src.models.lora_config import get_lora_config
from peft import get_peft_model, prepare_model_for_kbit_training

import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training Script")

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/sft")

    parser.add_argument("--max_length", type=int, default=256)  # reduced
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)

    parser.add_argument("--wandb_project", type=str, default="sft-training")
    parser.add_argument("--wandb_run_name", type=str, default="llama-sft")

    return parser.parse_args()


def load_sft_dataset(train_file, val_file):
    return load_dataset(
        "json",
        data_files={"train": train_file, "validation": val_file}
    )


def tokenize_fn(example, tokenizer, max_length):
    text = example["prompt"] + "\n" + example["response"]

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False
    )

    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main():
    args = parse_args()

    # ✅ W&B init
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args)
    )

    device = get_device()
    print(f"Using device: {device}")

    # ✅ 4-bit quantization (BIG MEMORY SAVE)
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4"
    # )

    model, tokenizer = load_base_model(
        args.model_name,
        device=device,
        use_fp16=True,
        quantization_config=None
    )

    # 🔥 VERY IMPORTANT for k-bit training
    # model = prepare_model_for_kbit_training(model)

    model.gradient_checkpointing_enable()

    lora_config = get_lora_config(
        r=args.lora_r,
        lora_alpha=args.lora_alpha
    )

    model = get_peft_model(model, lora_config)

    # 🔥 Fix for gradient checkpointing
    model.enable_input_require_grads()

    model.print_trainable_parameters()

    dataset = load_sft_dataset(args.train_file, args.val_file)

    tokenized = dataset.map(
        lambda x: tokenize_fn(x, tokenizer, args.max_length),
        remove_columns=dataset["train"].column_names,
        batched=False
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # use_fp16 = device == "cuda"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=False,
        bf16=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        report_to="wandb",   # ✅ logging enabled
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator
    )

    trainer.train(resume_from_checkpoint=False)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    wandb.finish()


if __name__ == "__main__":
    main()