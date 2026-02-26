# src/rl/train_ppo.py

import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from peft import PeftModel

from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from trl.experimental.ppo import PPOConfig, PPOTrainer
from trl import create_reference_model

from src.rl.reward import compute_reward


def parse_args():
    parser = argparse.ArgumentParser(description="PPO training for calibrated medical QA")
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model name or path"
    )
    parser.add_argument(
        "--sft_adapter",
        type=str,
        default="outputs/sft",
        help="Path to SFT adapter"
    )
    parser.add_argument(
        "--ppo_data",
        type=str,
        default="data/processed/ppo_train.jsonl",
        help="Path to PPO training data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/ppo",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="PPO batch size"
    )
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=2,
        help="PPO mini batch size"
    )
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=2,
        help="Number of PPO epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Logging interval"
    )
    parser.add_argument(
        "--value_model",
        type=str,
        default=None,
        help="Value model (defaults to distilbert on MPS, base_model on CUDA)"
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
        dtype = torch.float32  # MPS/CPU more stable

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    # Load LoRA adapter on top
    model = PeftModel.from_pretrained(model, args.sft_adapter)

    # Reference model (frozen copy, same base + adapter)
    ref_model = create_reference_model(model)
    ref_model = PeftModel.from_pretrained(ref_model, args.sft_adapter)

    # Value model selection
    if args.value_model:
        value_model_name = args.value_model
    elif device == "mps":
        value_model_name = "distilbert-base-uncased"
    else:
        value_model_name = args.base_model

    if device == "mps" or value_model_name == "distilbert-base-uncased":
        value_model = AutoModelForSequenceClassification.from_pretrained(
            value_model_name,
            num_labels=1,
        )
    else:
        value_model = AutoModelForSequenceClassification.from_pretrained(
            value_model_name,
            num_labels=1,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    # PPO configuration
    config = PPOConfig(
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        num_ppo_epochs=args.ppo_epochs,
        learning_rate=args.learning_rate,
    )

    # Load PPO training data
    dataset = load_dataset("json", data_files={"train": args.ppo_data})["train"]

    # Data collator for padding
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # Prepare PPOTrainer
    ppo_trainer = PPOTrainer(
        args=config,
        model=model,
        ref_model=ref_model,
        value_model=value_model,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        reward_model=value_model,
    )

    # Put model on device if not using device_map
    if device != "cuda":
        model.to(device)
        ref_model.to(device)

    # PPO training loop
    # PPO training loop
    for step, ex in enumerate(dataset):
        prompt = ex["prompt"]
        correct_option = ex["correct_option"]

        # 1) Tokenize prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        query_tensors = inputs.input_ids.to(model.device)

        # 2) Generate with the POLICY model (not PPOTrainer)
        with torch.no_grad():
            gen_outputs = model.generate(
                query_tensors,
                max_new_tokens=64,                 # tune this
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 3) Slice out only the newly generated tokens (the response)
        response_tensors = gen_outputs[:, query_tensors.shape[1]:]

        # 4) Decode response text (for your custom reward)
        response_text = tokenizer.decode(response_tensors[0], skip_special_tokens=True)

        # 5) Compute reward yourself
        reward, valid = compute_reward(response_text, correct_option)
        rewards = [torch.tensor(reward, dtype=torch.float32, device=model.device)]

        # 6) PPO step: pass query and response tensors
        ppo_trainer.step([query_tensors[0]], [response_tensors[0]], rewards)

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