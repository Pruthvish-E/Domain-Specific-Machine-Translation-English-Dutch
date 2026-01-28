import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

ENCDEC_BASE = RESULTS_DIR / "encdec_baseline_metrics.csv"
DEC_BASE = RESULTS_DIR / "deconly_baseline_metrics.csv"

ENCDEC_FT = RESULTS_DIR / "encdec_finetuned_metrics.csv"
DEC_LORA = RESULTS_DIR / "deconly_LoRA_finetuned_metrics.csv"


def load_metrics(path, model_name):
    df = pd.read_csv(path, index_col=0)
    df = df.T.reset_index()
    df.columns = ["domain", "bleu", "chrf", "avg_len_ratio"]
    df["model"] = model_name
    return df


def main():
    rows = []

    if ENCDEC_BASE.exists():
        rows.append(load_metrics(ENCDEC_BASE, "EncDec-Baseline"))

    if DEC_BASE.exists():
        rows.append(load_metrics(DEC_BASE, "DecOnly-Baseline"))

    if ENCDEC_FT.exists():
        rows.append(load_metrics(ENCDEC_FT, "EncDec-Finetuned"))

    if DEC_LORA.exists():
        rows.append(load_metrics(DEC_LORA, "DecOnly-LoRA"))

    all_metrics = pd.concat(rows, ignore_index=True)

    # Save master metrics table
    all_metrics.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    print("Saved metrics table to results/metrics_summary.csv")
    print(all_metrics)

    # -------------------------------
    # BLEU comparison plot
    # -------------------------------
    plt.figure(figsize=(9, 5))
    for model in all_metrics["model"].unique():
        sub = all_metrics[all_metrics["model"] == model]
        plt.plot(sub["domain"], sub["bleu"], marker="o", label=model)

    plt.title("BLEU Score Comparison Across Domains")
    plt.xlabel("Dataset")
    plt.ylabel("BLEU")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "bleu_comparison.png", dpi=200)
    plt.close()

    # -------------------------------
    # chrF comparison plot
    # -------------------------------
    plt.figure(figsize=(9, 5))
    for model in all_metrics["model"].unique():
        sub = all_metrics[all_metrics["model"] == model]
        plt.plot(sub["domain"], sub["chrf"], marker="o", label=model)

    plt.title("chrF++ Comparison Across Domains")
    plt.xlabel("Dataset")
    plt.ylabel("chrF++")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "chrf_comparison.png", dpi=200)
    plt.close()

    # -------------------------------
    # Domain adaptation effect plot
    # -------------------------------
    pivot = all_metrics.pivot(index="model", columns="domain", values="bleu")

    if "software" in pivot.columns and "flores" in pivot.columns:
        pivot["domain_gain"] = pivot["software"] - pivot["flores"]

        plt.figure(figsize=(8, 5))
        plt.bar(pivot.index, pivot["domain_gain"])
        plt.axhline(0)
        plt.title("Domain Shift Effect (BLEU: Software – Flores)")
        plt.ylabel("BLEU Gain")
        plt.xticks(rotation=15)
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "domain_shift.png", dpi=200)
        plt.close()

    print("Saved plots:")
    print("- results/bleu_comparison.png")
    print("- results/chrf_comparison.png")
    print("- results/domain_shift.png")


if __name__ == "__main__":
    main()
