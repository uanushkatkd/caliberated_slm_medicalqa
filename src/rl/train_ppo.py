# src/rl/train_ppo.py

import argparse
import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from peft import PeftModel

from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

from src.rl.reward import compute_reward


def parse_args():
    parser = argparse.ArgumentParser(description="PPO training for calibrated medical QA")

    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--sft_adapter", type=str, default="outputs/sft")
    parser.add_argument("--ppo_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/ppo")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=2)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=10)

    # ✅ W&B args
    parser.add_argument("--wandb_project", type=str, default="ppo-training")
    parser.add_argument("--wandb_run_name", type=str, default="ppo-run")

    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()

    # ✅ Init W&B
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args)
    )

    device = get_device()
    print(f"Using device: {device}")

    dtype = torch.float16 if device == "cuda" else torch.float32

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy model
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    model.pretrained_model = PeftModel.from_pretrained(
        model.pretrained_model,
        args.sft_adapter,
    )

    # Reference model (frozen)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    ref_model.pretrained_model = PeftModel.from_pretrained(
        ref_model.pretrained_model,
        args.sft_adapter
    )

    # Move if CPU/MPS
    if device != "cuda":
        model.to(device)
        ref_model.to(device)

    config = PPOConfig(
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        learning_rate=args.learning_rate,
    )

    dataset = load_dataset("json", data_files={"train": args.ppo_data})["train"]

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 🚀 TRAIN LOOP
    # for step, ex in enumerate(dataset):
    #     prompt = ex["prompt"]
    #     correct_option = ex["correct_option"]

    #     query_tensor = tokenizer(
    #         prompt,
    #         return_tensors="pt",
    #         truncation=True,
    #     ).input_ids.to(device)[0]

    #     response_tensor = ppo_trainer.generate(
    #         [query_tensor],
    #         max_new_tokens=64,
    #         do_sample=True,
    #         top_p=0.9,
    #         temperature=0.7,
    #         pad_token_id=tokenizer.eos_token_id,
    #     )[0]

    #     response_text = tokenizer.decode(response_tensor, skip_special_tokens=True)

    #     reward, valid = compute_reward(response_text, correct_option)
    #     reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device)

    #     stats = ppo_trainer.step(
    #         [query_tensor],
    #         [response_tensor],
    #         [reward_tensor]
    #     )

    #     # ✅ W&B logging
    #     log_data = {
    #         "step": step,
    #         "reward": reward,
    #         "valid": int(valid),
    #     }

    #     # Add PPO stats if available
    #     if stats is not None:
    #         log_data.update({
    #             k: v for k, v in stats.items() if isinstance(v, (int, float))
    #         })

    #     wandb.log(log_data)

    #     if step % args.log_interval == 0:
    #         print(f"Step {step} | reward={reward:.3f} | valid={valid}")

    #     if step >= args.max_steps:
    #         break
    
    batch_queries = []
    batch_responses = []
    batch_rewards = []

    for step, ex in enumerate(dataset):
        prompt = ex["prompt"]
        correct_option = ex["correct_option"]

        query_tensor = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).input_ids.to(device)[0]

        response_tensor = ppo_trainer.generate(
            [query_tensor],
            max_new_tokens=64,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )[0]

        response_text = tokenizer.decode(response_tensor, skip_special_tokens=True)

        reward, valid = compute_reward(response_text, correct_option)

        batch_queries.append(query_tensor)
        batch_responses.append(response_tensor)
        batch_rewards.append(torch.tensor(reward, dtype=torch.float32).to(device))

        # 🔥 When batch is ready → PPO step
        if len(batch_queries) == args.batch_size:
            stats = ppo_trainer.step(
                batch_queries,
                batch_responses,
                batch_rewards
            )

            # Logging
            wandb.log({
                "step": step,
                "avg_reward": sum([r.item() for r in batch_rewards]) / len(batch_rewards),
                "valid_ratio": sum([1 for r in batch_rewards]) / len(batch_rewards),
                **{k: v for k, v in (stats or {}).items() if isinstance(v, (int, float))}
            })

            # Reset batch
            batch_queries, batch_responses, batch_rewards = [], [], []

        if step % args.log_interval == 0:
            print(f"Step {step} | last_reward={reward:.3f} | valid={valid}")

        if step >= args.max_steps:
            break

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    wandb.finish()

    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()