# src/models/load_model.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_base_model(
    model_name: str,
    device: str = None,
    use_fp16: bool = True,
    quantization_config: BitsAndBytesConfig = None,
):
    """
    Loads base causal LM and tokenizer with optional quantization.

    Args:
        model_name (str): HuggingFace model id
        device (str): Device to load model on
        use_fp16 (bool): Use fp16 weights (CUDA only)
        quantization_config (BitsAndBytesConfig): 4-bit / 8-bit config

    Returns:
        model, tokenizer
    """

    if device is None:
        device = get_device()

    # 🔹 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 🔹 Dtype
    if device == "cuda":
        # dtype = torch.float16 if use_fp16 else torch.float32
        dtype = torch.bfloat16
    else:
        dtype = torch.float32  # safer for CPU/MPS

    # 🔥 CASE 1: Quantized model (only CUDA)
    if quantization_config is not None and device == "cuda":
        print("Loading model with quantization...")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=None,
            trust_remote_code=True,
        )

    # 🔹 CASE 2: Standard GPU
    elif device == "cuda":
        print("Loading model on CUDA...")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    # 🔹 CASE 3: CPU / MPS
    else:
        print(f"Loading model on {device}...")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)

    # 🔥 Important for training stability
    model.config.use_cache = False

    return model, tokenizer