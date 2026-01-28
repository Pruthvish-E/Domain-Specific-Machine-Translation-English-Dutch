import sacrebleu
from statistics import mean


def compute_metrics(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    chrf = sacrebleu.corpus_chrf(preds, [refs])

    length_ratios = [
        len(p.split()) / max(1, len(r.split()))
        for p, r in zip(preds, refs)
    ]

    return {
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 3),
        "avg_len_ratio": round(mean(length_ratios), 3)
    }
