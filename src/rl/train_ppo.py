import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

from src.models.load_model import load_base_model
from src.models.lora_config import get_lora_config
from src.rl.reward import compute_reward

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
SFT_ADAPTER = "outputs/sft"
PPO_DATA = "data/processed/ppo_train.jsonl"
OUT_DIR = "outputs/ppo"

def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base + value head
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Load SFT LoRA adapter
    model = PeftModel.from_pretrained(model, SFT_ADAPTER)

    # Reference model
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16
    )
    ref_model = PeftModel.from_pretrained(ref_model, SFT_ADAPTER)

    config = PPOConfig(
        batch_size=8,
        mini_batch_size=4,
        ppo_epochs=2,
        learning_rate=1e-5,
        log_with=None
    )

    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    data = load_dataset("json", data_files={"train": PPO_DATA})["train"]

    for step, ex in enumerate(data):
        prompt = ex["prompt"]
        correct_option = ex["correct_option"]

        query = tokenizer(prompt, return_tensors="pt").to(model.device)

        response_tensors = ppo_trainer.generate(
            **query,
            max_new_tokens=50,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
        )

        response_text = tokenizer.decode(response_tensors[0], skip_special_tokens=True)

        reward, valid = compute_reward(response_text, correct_option)

        rewards = [torch.tensor(reward, device=model.device)]

        ppo_trainer.step([query["input_ids"][0]], [response_tensors[0]], rewards)

        if step % 50 == 0:
            print(f"Step {step}, reward={reward}, valid={valid}")

        if step >= 1000:  # safety stop for Kaggle
            break

    # Save calibrated adapter
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

if __name__ == "__main__":
    main()
