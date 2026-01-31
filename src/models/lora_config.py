# src/models/lora_config.py

from peft import LoraConfig, TaskType


def get_lora_config(
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
):
    """
    Returns a LoRA configuration optimized for Kaggle GPUs.

    Args:
        r (int): LoRA rank (8 is a good Kaggle-safe default)
        lora_alpha (int): scaling factor
        lora_dropout (float): dropout for regularization

    Returns:
        LoraConfig
    """

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "v_proj"
        ],
    )
