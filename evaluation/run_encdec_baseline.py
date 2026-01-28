import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import pandas as pd
from utils import compute_metrics

MODEL_NAME = "Helsinki-NLP/opus-mt-en-nl"

SOFTWARE_DATASET = "data/processed/software_mt"
FLORES_DATASET = "data/processed/flores_en_nl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_eval(dataset_path, tag):
    ds = load_from_disk(dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    preds, refs, srcs = [], [], []

    for sample in tqdm(ds, desc=f"Evaluating {tag}"):
        src = sample["translation"]["en"]
        ref = sample["translation"]["nl"]

        inputs = tokenizer(src, return_tensors="pt", truncation=True).to(DEVICE)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4
            )

        pred = tokenizer.decode(out[0], skip_special_tokens=True)

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

    df.to_csv(f"results/encdec_{tag}_predictions.csv", index=False)

    return metrics


if __name__ == "__main__":
    all_metrics = {}
    all_metrics["software"] = run_eval(SOFTWARE_DATASET, "software")
    all_metrics["flores"] = run_eval(FLORES_DATASET, "flores")

    pd.DataFrame(all_metrics).to_csv("results/encdec_baseline_metrics.csv")
