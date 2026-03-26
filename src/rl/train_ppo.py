# src/rl/train_ppo.py

import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from peft import PeftModel

from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

from src.rl.reward import compute_reward


def parse_args():
    parser = argparse.ArgumentParser(description="PPO training for calibrated medical QA")

    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--sft_adapter",
        type=str,
        default="outputs/sft",
        help="Path to SFT adapter",
    )
    parser.add_argument(
        "--ppo_data",
        type=str,
        default="data/processed/ppo_train.jsonl",
        help="Path to PPO training data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/ppo",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="PPO batch size",
    )
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=2,
        help="PPO mini batch size",
    )
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=2,
        help="Number of PPO epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Logging interval",
    )

    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Use safer dtype on MPS, fp16 on CUDA
    if device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model WITH VALUE HEAD (important for old PPO)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    # Load LoRA adapter on top
    model.pretrained_model = PeftModel.from_pretrained(
    model.pretrained_model,
    args.sft_adapter,
)

    # Reference model (frozen copy)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    ref_model.pretrained_model = PeftModel.from_pretrained(ref_model.pretrained_model, args.sft_adapter)

    # Move to device if needed
    if device != "cuda":
        model.to(device)
        ref_model.to(device)

    # PPO configuration
    config = PPOConfig(
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        learning_rate=args.learning_rate,
    )

    # Load PPO training data
    dataset = load_dataset("json", data_files={"train": args.ppo_data})["train"]

    # Data collator (not strictly needed for manual loop, but fine to keep)
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # Prepare PPOTrainer (OLD API: no reward_model, no value_model needed)
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # PPO training loop (OLD STYLE)
    for step, ex in enumerate(dataset):
        prompt = ex["prompt"]
        correct_option = ex["correct_option"]

        query_tensors = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        ).input_ids.to(device)

        query_tensor = query_tensors[0]  # shape: (seq_len,)

        response_tensors = ppo_trainer.generate(
            [query_tensor],
            max_new_tokens=64,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

        response_tensor = response_tensors[0]

        response_text = tokenizer.decode(response_tensor, skip_special_tokens=True)

        reward, valid = compute_reward(response_text, correct_option)
        rewards = [torch.tensor(reward, dtype=torch.float32, device=device)]

        ppo_trainer.step([query_tensor], [response_tensor], rewards)

        if step % args.log_interval == 0:
            print(f"Step {step}, reward={reward}, valid={valid}")

        if step >= args.max_steps:
            break
        # Save final model + tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()