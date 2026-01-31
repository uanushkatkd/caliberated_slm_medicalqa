# src/models/load_model.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base_model(
    model_name: str,
    device_map: str = "auto",
    use_fp16: bool = True,
):
    """
    Loads base causal LM and tokenizer for SFT / PPO training.

    Args:
        model_name (str): HuggingFace model id
        device_map (str): "auto" recommended for Kaggle
        use_fp16 (bool): enable fp16 for memory efficiency

    Returns:
        model, tokenizer
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side="right",
        
    )

    # Important for causal LM fine-tuning
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if use_fp16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
     )

    # Disable cache for training stability (important for PPO)
    model.config.use_cache = False

    return model, tokenizer
