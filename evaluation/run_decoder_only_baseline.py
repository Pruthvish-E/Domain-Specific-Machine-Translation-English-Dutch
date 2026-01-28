import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
from utils import compute_metrics

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

SOFTWARE_DATASET = "data/processed/software_mt"
FLORES_DATASET = "data/processed/flores_en_nl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_prompt(text):
    return f"""### Instruction:
Translate the following software-related text from English to Dutch.

### Input:
{text}

### Response:
"""


def extract_translation(output, prompt):
    return output.replace(prompt, "").strip()


def run_eval(dataset_path, tag):
    ds = load_from_disk(dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    preds, refs, srcs = [], [], []

    for sample in tqdm(ds, desc=f"Evaluating {tag}"):
        src = sample["translation"]["en"]
        ref = sample["translation"]["nl"]

        prompt = make_prompt(src)

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.2,
                do_sample=False
            )

        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = extract_translation(decoded, prompt)

        preds.append(pred)
        refs.append(ref)
        srcs.append(src)

    metrics = compute_metrics(preds, refs)
    print(tag, metrics)

    df = pd.DataFrame({
        "source": srcs,
        "reference": refs,
        "prediction": preds
    })

    df.to_csv(f"results/deconly_{tag}_predictions.csv", index=False)

    return metrics


if __name__ == "__main__":
    all_metrics = {}
    all_metrics["software"] = run_eval(SOFTWARE_DATASET, "software")
    all_metrics["flores"] = run_eval(FLORES_DATASET, "flores")

    pd.DataFrame(all_metrics).to_csv("results/deconly_baseline_metrics.csv")
