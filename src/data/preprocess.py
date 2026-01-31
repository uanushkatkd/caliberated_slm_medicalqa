import json
from pathlib import Path
from src.utils.paths import PROCESSED_DATA_DIR


def format_question(sample):
    opts = sample["options"]
    return (
        f"{sample['question']}\n\n"
        f"A. {opts['A']}\n"
        f"B. {opts['B']}\n"
        f"C. {opts['C']}\n"
        f"D. {opts['D']}"
    )


def create_sft_dataset(samples, output_path):
    """
    SFT: model learns to answer (no confidence yet).
    """
    output_path = PROCESSED_DATA_DIR / output_path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for s in samples:
            record = {
                "prompt": format_question(s),
                "response": f"The correct answer is {s['correct_option']}."
            }
            f.write(json.dumps(record) + "\n")


def create_ppo_dataset(samples, output_path):
    """
    PPO: no answers provided, model must decide + express confidence.
    """
    output_path = PROCESSED_DATA_DIR / output_path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for s in samples:
            record = {
                "prompt": format_question(s),
                "correct_option": s["correct_option"]  # used only by reward fn
            }
            f.write(json.dumps(record) + "\n")
