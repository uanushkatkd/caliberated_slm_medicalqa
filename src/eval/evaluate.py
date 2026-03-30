# src/eval/evaluate.py

import argparse
import json
import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.eval.metrics import accuracy, expected_calibration_error, auroc
from src.utils.parsing import parse_answer_and_confidence
from src.eval.calibration import (
    plot_reliability_diagram,
    plot_confidence_histogram
)
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--val_file", type=str, required=True)

    parser.add_argument("--sft_adapter", type=str, required=True)
    parser.add_argument("--ppo_adapter", type=str, required=True)

    parser.add_argument("--out_dir", type=str, default="outputs/eval")

    parser.add_argument("--wandb_project", type=str, default=None)

    return parser.parse_args()


def eval_model(base_model, adapter_path, val_file, out_path):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    data = load_dataset("json", data_files={"val": val_file})["val"]

    y_true, y_pred, confidences, correct_flags = [], [], [], []

    for ex in data:
        prompt = ex["prompt"]
        gt = ex["correct_option"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)

        text = tokenizer.decode(out[0], skip_special_tokens=True)

        pred, conf, valid = parse_answer_and_confidence(text)

        if not valid or conf is None or pred is None:
            continue

        y_true.append(gt)
        y_pred.append(pred)
        confidences.append(conf)
        correct_flags.append(1 if pred == gt else 0)

    metrics = {
    "accuracy": accuracy(y_true, y_pred),
    "ece": expected_calibration_error(confidences, correct_flags),
    "auroc": auroc(confidences, correct_flags),
    "n": len(y_true)
    }

    print(out_path, metrics)

    # ✅ Save metrics
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ✅ Save plots
    rel_path = out_path.replace(".json", "_reliability.png")
    hist_path = out_path.replace(".json", "_hist.png")

    plot_reliability_diagram(confidences, correct_flags, rel_path)
    plot_confidence_histogram(confidences, hist_path)

    return metrics, rel_path, hist_path


def main():
    args = parse_args()

    if args.wandb_project:
        wandb.init(project=args.wandb_project, name="eval")
    sft_metrics, sft_rel, sft_hist = eval_model(
    args.base_model,
    args.sft_adapter,
    args.val_file,
    f"{args.out_dir}/sft_metrics.json"
    )

    ppo_metrics, ppo_rel, ppo_hist = eval_model(
        args.base_model,
        args.ppo_adapter,
        args.val_file,
        f"{args.out_dir}/ppo_metrics.json"
    )

    if args.wandb_project:
        wandb.log({
            "sft_accuracy": sft_metrics["accuracy"],
            "ppo_accuracy": ppo_metrics["accuracy"],
            "sft_ece": sft_metrics["ece"],
            "ppo_ece": ppo_metrics["ece"],

            # ✅ Upload plots
            "sft_reliability": wandb.Image(sft_rel),
            "ppo_reliability": wandb.Image(ppo_rel),
            "sft_conf_hist": wandb.Image(sft_hist),
            "ppo_conf_hist": wandb.Image(ppo_hist),
        })

if __name__ == "__main__":
    main()