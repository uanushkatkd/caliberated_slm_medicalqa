# src/eval/eval_baseline.py

import argparse
import json
import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.eval.metrics import accuracy, expected_calibration_error, auroc
from src.utils.parsing import parse_answer_and_confidence


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--out_path", type=str, default="outputs/eval/baseline.json")

    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default="baseline-eval")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    data = load_dataset("json", data_files={"val": args.val_file})["val"]

    y_true, y_pred, confidences, correct_flags = [], [], [], []

    for ex in data:
        prompt = ex["prompt"] + "\nAnswer:"
        gt = ex["response"].strip()[-2]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10)

        text = tokenizer.decode(out[0], skip_special_tokens=True)

        pred, _, valid = parse_answer_and_confidence(text)

        if pred is None:
            continue

        y_true.append(gt)
        y_pred.append(pred)

        conf = 0.5  # baseline
        confidences.append(conf)
        correct_flags.append(1 if pred == gt else 0)

    acc = accuracy(y_true, y_pred)
    ece = expected_calibration_error(confidences, correct_flags)
    auc = auroc(confidences, correct_flags)

    metrics = {"accuracy": acc, "ece": ece, "auroc": auc, "n": len(y_true)}

    print(metrics)

    with open(args.out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    if args.wandb_project:
        wandb.log(metrics)
        wandb.finish()


if __name__ == "__main__":
    main()