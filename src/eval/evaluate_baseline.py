import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.eval.metrics import accuracy, expected_calibration_error, auroc
from src.utils.parsing import parse_answer_and_confidence

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "outputs/sft"
VAL_FILE = "data/processed/sft_val.jsonl"
OUT_PATH = "outputs/eval/baseline_metrics.json"

def main():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    data = load_dataset("json", data_files={"val": VAL_FILE})["val"]

    y_true = []
    y_pred = []
    confidences = []
    correct_flags = []

    for ex in data:
        prompt = ex["prompt"] + "\nAnswer:"
        gt = ex["response"].strip()[-2]  # assumes "The correct answer is C."
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10)
        text = tokenizer.decode(out[0], skip_special_tokens=True)

        # Parse answer only (no confidence yet)
        pred, _, valid = parse_answer_and_confidence(text)

        if pred is None:
            continue

        y_true.append(gt)
        y_pred.append(pred)

        # Baseline confidence proxy: constant 0.5
        conf = 0.5
        confidences.append(conf)
        correct_flags.append(1 if pred == gt else 0)

    acc = accuracy(y_true, y_pred)
    ece = expected_calibration_error(confidences, correct_flags)
    auc = auroc(confidences, correct_flags)

    metrics = {
        "accuracy": acc,
        "ece": ece,
        "auroc": auc,
        "n": len(y_true)
    }

    print(metrics)

    with open(OUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
