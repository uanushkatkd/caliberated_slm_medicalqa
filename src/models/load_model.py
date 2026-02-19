# src/models/load_model.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_base_model(
    model_name: str,
    device: str = None,
    use_fp16: bool = True,
):
    """
    Loads base causal LM and tokenizer for SFT / PPO training.

    Args:
        model_name (str): HuggingFace model id
        device (str): Device to load model on. Auto-detects if None.
        use_fp16 (bool): Load model weights in fp16 to save memory.

    Returns:
        model, tokenizer
    """
    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # fp16 weights save memory; operations still run in fp32 on MPS
    dtype = torch.float16 if use_fp16 else torch.float32

    if device in ("mps", "cpu"):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    model.config.use_cache = False

    return model, tokenizer
