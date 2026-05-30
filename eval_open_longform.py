"""
eval_open_longform.py — Open-metric evaluation (ROUGE-L, BERTScore) for
long-form QA outputs on the Biographies and HaluSum benchmarks.

Usage:
    python eval_open_longform.py \
        --file_path output/Ebio_Tbio_N100_qwen2.5-3b_rcd_test_100.json \
        --eval_data bio \
        --model qwen2.5-3b \
        --decoding_method rcd \
        --out_dir evaluation_results/

    python eval_open_longform.py \
        --file_path output/Ehalu_sum_Thalu_sum_N400_qwen2.5-3b_knn_lm_test_100.json \
        --eval_data halu_sum \
        --model qwen2.5-3b \
        --decoding_method knn_lm \
        --out_dir evaluation_results/
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from rouge_score import rouge_scorer as rouge_scorer_lib
from bert_score import score as bert_score_fn

# ---------------------------------------------------------------------------
# Rouge-L helper
# ---------------------------------------------------------------------------
_rouge = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(pred: str, ref: str) -> float:
    return _rouge.score(ref, pred)["rougeL"].fmeasure


# ---------------------------------------------------------------------------
# BERTScore helper (batched for speed)
# ---------------------------------------------------------------------------
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"


def bertscore_batch(preds: list[str], refs: list[str]) -> list[float]:
    P, R, F1 = bert_score_fn(
        preds,
        refs,
        model_type=BERTSCORE_MODEL,
        lang="en",
        rescale_with_baseline=True,
        verbose=False,
    )
    return [float(f) for f in F1]


# ---------------------------------------------------------------------------
# Bio reference loader
# ---------------------------------------------------------------------------
def _normalize_name(name: str) -> str:
    if "(" in name:
        name = name.split("(")[0]
    return name.strip()


def load_bio_references(article_path: str = "data/article_200.json") -> dict[str, str]:
    with open(article_path) as f:
        raw = json.load(f)
    return {_normalize_name(k): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# HaluSum reference loader (uses HuggingFace dataset)
# ---------------------------------------------------------------------------
def load_halu_sum_references(split: str = "test", max_samples: int = 100) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", "summarization")["data"]

    refs = []
    for sample in ds:
        if len(sample["document"]) > 1500:
            continue
        refs.append(sample["right_summary"])

    if split == "test":
        refs = refs[100:]
    elif split == "train":
        refs = refs[:100]
    return refs[:max_samples]


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate(
    file_path: str,
    eval_data: str,
    model: str,
    decoding_method: str,
    out_dir: str,
    bertscore_model: str = BERTSCORE_MODEL,
) -> dict:
    response_data = json.load(open(file_path))
    n = len(response_data)

    preds: list[str] = []
    refs: list[str] = []

    if eval_data == "bio":
        article_dict = load_bio_references()
        for sample in response_data:
            person = _normalize_name(sample["question"])
            if person not in article_dict:
                continue
            preds.append(sample["generated_answer"])
            refs.append(article_dict[person])

    elif eval_data == "halu_sum":
        ref_list = load_halu_sum_references(split="test", max_samples=n)
        for i, sample in enumerate(response_data):
            if i >= len(ref_list):
                break
            preds.append(sample["generated_answer"])
            refs.append(ref_list[i])

    else:
        raise ValueError(f"Unsupported eval_data: {eval_data}")

    if not preds:
        print("[Warning] No matched samples found.")
        return {}

    print(f"Evaluating {len(preds)} samples for {eval_data}/{model}/{decoding_method}...")

    # ROUGE-L
    rouge_scores = [rouge_l(p, r) for p, r in zip(preds, refs)]
    mean_rouge = float(np.mean(rouge_scores)) * 100

    # BERTScore (batched)
    bert_scores = bertscore_batch(preds, refs)
    mean_bert = float(np.mean(bert_scores)) * 100

    result = {
        "eval_data": eval_data,
        "model": model,
        "decoding_method": decoding_method,
        "file_path": str(file_path),
        "n_samples": len(preds),
        "rougeL": round(mean_rouge, 2),
        "bertscore_f1": round(mean_bert, 2),
        "timestamp": round(time.time()),
    }

    print(f"  RougeL={mean_rouge:.2f}  BERTScore={mean_bert:.2f}")

    os.makedirs(out_dir, exist_ok=True)
    fname = f"{eval_data}_{model}_{decoding_method}_open_metrics.json"
    out_path = os.path.join(out_dir, fname)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {out_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Open-metric eval for bio/halu_sum.")
    parser.add_argument("--file_path", required=True, help="Path to generated output JSON.")
    parser.add_argument("--eval_data", required=True, choices=["bio", "halu_sum"])
    parser.add_argument("--model", required=True, help="Base model name (e.g. qwen2.5-3b).")
    parser.add_argument("--decoding_method", required=True, help="Method name (e.g. greedy, rcd).")
    parser.add_argument("--out_dir", default="evaluation_results/open_metrics",
                        help="Directory to save results.")
    parser.add_argument("--bertscore_model", default=BERTSCORE_MODEL)
    args = parser.parse_args()

    evaluate(
        file_path=args.file_path,
        eval_data=args.eval_data,
        model=args.model,
        decoding_method=args.decoding_method,
        out_dir=args.out_dir,
        bertscore_model=args.bertscore_model,
    )


if __name__ == "__main__":
    main()
