# src/sft/train_sft_unsloth.py

import argparse
from datasets import load_dataset

from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
import wandb



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="unsloth/llama-3-1b-bnb-4bit")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/sft")

    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)

    return parser.parse_args()


def formatting_func(example):
    return example["prompt"] + "\n" + example["response"]



def main():
    args = parse_args()

    wandb.init(
        project="sft-kaggle",
        name="unsloth-sft-run",
        config=vars(args),
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=formatting_func,
        max_seq_length=args.max_seq_length,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=20,
            save_steps=200,
            output_dir=args.output_dir,
            report_to="wandb",          # ✅ enable logging
            run_name="unsloth-sft-run", # ✅ visible in dashboard
        ),
    )

    trainer.train()

    wandb.finish()
if __name__ == "__main__":
    main()