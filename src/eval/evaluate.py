import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.eval.metrics import accuracy, expected_calibration_error, auroc
from src.utils.parsing import parse_answer_and_confidence

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
VAL_FILE = "data/processed/ppo_train.jsonl"

def eval_model(adapter_path, out_path):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    data = load_dataset("json", data_files={"val": VAL_FILE})["val"]

    y_true = []
    y_pred = []
    confidences = []
    correct_flags = []

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

    acc = accuracy(y_true, y_pred)
    ece = expected_calibration_error(confidences, correct_flags)
    auc = auroc(confidences, correct_flags)

    metrics = {
        "accuracy": acc,
        "ece": ece,
        "auroc": auc,
        "n": len(y_true)
    }

    print(out_path, metrics)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

def main():
    eval_model("outputs/sft", "outputs/eval/sft_metrics.json")
    eval_model("outputs/ppo", "outputs/eval/ppo_metrics.json")

if __name__ == "__main__":
    main()
