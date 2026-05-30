"""
run.py — Unified inference + evaluation CLI for all DecodeHub decoding methods.

Supported decoding methods:
    greedy      Standard greedy decoding (HF .generate)
    rcd         RAD: retrieval-augmented contrastive decoding (cosine FAISS)
    knn_lm      kNN-LM probability interpolation (L2 FAISS datastore)
    instructive Instructive decoding (contrastive with noisy prompt)
    cad         Context-aware decoding (contrastive with no-context prompt)
    dola         DoLa (divergence of layers)
    icl         In-context learning (greedy + few-shot prompt, no retrieval)
    KAPING      KAPING: knowledge-graph triplets as few-shot context
    kNN-ICL     kNN-ICL: retrieved Q&A pairs as few-shot context

Usage examples
--------------
# Single run — greedy on TruthfulQA:
python run.py --base_model llama3.1-8b --decoding_method greedy \
              --eval_data truthful_qa

# RAD sweep:
python run.py --base_model llama3.1-8b --decoding_method rcd \
              --eval_data wiki --train_data truthful_qa \
              --configs_json '[{"shaping_mode":"linear","alpha":0.5,"sim_threshold":0.7}]'

# kNN-LM:
python run.py --base_model llama3.1-8b --decoding_method knn_lm \
              --eval_data truthful_qa --train_data truthful_qa \
              --configs_json '[{"alpha":0.5}]'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import torch
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from utils.config import eval_result_dir, get_model_path, output_dir
from utils.evaluation import eval_single_file, load_data_eval
from utils.generation import AllDecoding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_to_str(cfg: dict) -> str:
    return "--".join(f"{k}__{v}" for k, v in sorted(cfg.items()))


def _output_path(
    output_dir: str,
    eval_data:  str,
    train_data: str,
    num_train:  int,
    base_model: str,
    method:     str,
    split:      str,
    n_samples:  int,
    suffix:     str = "",
) -> str:
    name = f"E{eval_data}_T{train_data}_N{num_train}_{base_model}_{method}_{split}_{n_samples}"
    if suffix:
        name += f"_{suffix}"
    return str(Path(output_dir) / f"{name}.json")


def _save_output(path: str, questions: list[str], responses: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = [
        {"question": q, "generated_answer": r, "question_index": i}
        for i, (q, r) in enumerate(zip(questions, responses))
    ]
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Saved] {path}")


def _save_timing(path: str, timing: dict) -> None:
    timing_path = path.replace(".json", "_timing.json")
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)
    print(f"[Timing saved] {timing_path}")


def _save_eval_result(result: dict, args: argparse.Namespace) -> None:
    os.makedirs(eval_result_dir, exist_ok=True)
    ts  = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")
    fn  = (
        f"{args.decoding_method}__{args.eval_data}__{args.base_model}"
        f"__{args.data_split}__{args.max_sample_num}__{ts}.json"
    )
    path = os.path.join(eval_result_dir, fn)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Eval saved] {path}")


def _get_cpu_rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Default config sweeps per (method, eval_data)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIGS: dict[tuple[str, str], list[dict]] = {
    ("rcd", "truthful_qa"): [
        {"shaping_mode": "linear",  "alpha": 0.5, "sim_threshold": 0.7, "agg_mode": "weighted"},
        {"shaping_mode": "linear",  "alpha": 0.5, "sim_threshold": 0.7, "exact_match": "ignore"},
        {"shaping_mode": "greedy"},
    ],
    ("rcd", "wiki"): [
        {"shaping_mode": "linear",  "alpha": 0.5, "sim_threshold": 0.7, "agg_mode": "weighted"},
        {"shaping_mode": "greedy"},
    ],
    ("rcd", "alpaca"): [
        {"shaping_mode": "linear",  "alpha": 0.5, "sim_threshold": 0.8, "agg_mode": "weighted"},
        {"shaping_mode": "greedy"},
    ],
}

_GREEDY_CONFIG: list[dict] = [{}]


def _get_default_configs(method: str, eval_data: str) -> list[dict]:
    key = (method, eval_data)
    return _DEFAULT_CONFIGS.get(key, _GREEDY_CONFIG)


# ---------------------------------------------------------------------------
# Per-method decoding config builders
# ---------------------------------------------------------------------------

def _build_rcd_cfg(args: argparse.Namespace, extra: dict) -> dict:
    base = {
        "model_name":      args.base_model,
        "train_data":      args.train_data,
        "embed_model_name": args.embed_model_name,
        "num_train":       args.num_train,
        "top_n":           None,
        "chunk_size":      8,
        "shaping_mode":    "linear",
        "sim_threshold":   0.7,
        "agg_mode":        "weighted",
        "exact_match":     "",
        "alpha":           0.5,
    }
    base.update(extra)
    return base


def _build_instructive_cfg(args: argparse.Namespace, extra: dict) -> dict:
    return {"eta": args.eta, **extra}


def _build_cad_cfg(args: argparse.Namespace, extra: dict) -> dict:
    return {"eta": args.eta, **extra}


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def run_single(
    args: argparse.Namespace,
    decoding_config: dict,
    suffix: str = "",
) -> Optional[dict]:
    """Run inference for one config, optionally evaluate, return eval result."""
    out_path = _output_path(
        output_dir   = output_dir,
        eval_data    = args.eval_data,
        train_data   = args.train_data,
        num_train    = args.num_train,
        base_model   = args.base_model,
        method       = args.decoding_method,
        split        = args.data_split,
        n_samples    = args.max_sample_num,
        suffix       = suffix,
    )
    args.eval_data_path = out_path
    args.decoding_config = decoding_config

    # Load data
    data      = load_data_eval(args)
    questions = [d["question"] for d in data]
    answers   = (
        [d.get("correct_answers", [""])[0] if isinstance(d.get("correct_answers"), list)
         else d.get("answer", "") for d in data]
        if args.eval_data.startswith("halu")
        else None
    )

    print(f"[Inference] method={args.decoding_method} config={decoding_config or '{}'}")
    model = AllDecoding(
        model_path       = args.model_path,
        decoding_method  = args.decoding_method,
        max_new_tokens   = args.max_new_tokens,
        decoding_config  = decoding_config,
        prompt_key       = args.prompt_key,
        noisy_prompt_key = args.noisy_prompt_key,
        train_data       = args.train_data,
        device           = args.device,
        embedding_model  = args.embed_model_name,
    )

    stats_path = (
        out_path.replace(".json", "_stats.pkl")
        if getattr(args, "save_decoding_stats", False) else None
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cpu_before = _get_cpu_rss_mb()

    t0        = time.perf_counter()
    responses = model.inference_on_dataset(
        questions, answers=answers, batch_size=args.batch_size, stats_path=stats_path
    )
    elapsed   = time.perf_counter() - t0

    peak_gpu_mb  = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
    delta_cpu_mb = _get_cpu_rss_mb() - cpu_before

    n = len(responses)
    total_tokens = sum(len(r.split()) for r in responses)
    timing = {
        "method":              args.decoding_method,
        "eval_data":           args.eval_data,
        "train_data":          args.train_data,
        "base_model":          args.base_model,
        "n_samples":           n,
        "total_sec":           round(elapsed, 3),
        "mean_sec_per_sample": round(elapsed / n, 4) if n else 0,
        "approx_tokens":       total_tokens,
        "approx_tokens_per_sec": round(total_tokens / elapsed, 2) if elapsed > 0 else 0,
        "peak_gpu_mb":         round(peak_gpu_mb, 1),
        "delta_cpu_mb":        round(delta_cpu_mb, 1),
    }
    print(
        f"[Done] {elapsed:.2f}s total | {timing['mean_sec_per_sample']}s/sample "
        f"| ~{timing['approx_tokens_per_sec']} tok/s"
    )

    _save_output(out_path, questions, responses)

    if args.run_only:
        _save_timing(out_path, timing)
        return None

    print(f"[Eval] {args.decoding_method} on {args.eval_data}")
    result = eval_single_file(out_path, args=args)
    _save_eval_result(result, args)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DecodeHub unified inference + evaluation.")

    # Model
    p.add_argument("--base_model",       required=True,  help="Model alias from config.yaml.")
    p.add_argument("--device",           default="auto")
    p.add_argument("--max_new_tokens",   type=int, default=256)

    # Data
    p.add_argument("--eval_data",        default="truthful_qa",
                   help="Evaluation dataset key (truthful_qa, wiki, alpaca, gsm8k, "
                        "halu_qa, halu_dia, halu_sum, bio, faith_unanswerable, "
                        "faith_inconsistent, faith_counterfactual, precisewiki).")
    p.add_argument("--train_data",       default="",
                   help="Training dataset used for kNN-LM/RAD datastore precompute.")
    p.add_argument("--data_split",       default="test")
    p.add_argument("--max_sample_num",   type=int, default=100)
    p.add_argument("--num_train",        type=int, default=100,
                   help="Number of training samples used in the datastore.")

    # Method
    p.add_argument("--decoding_method",  default="greedy",
                   choices=["greedy", "knn_lm", "rcd", "instructive",
                             "cad", "dola", "icl", "KAPING", "kNN-ICL"],
                   help=(
                       "Decoding method. "
                       "--- DECODING-TIME (logit intervention) --- "
                       "greedy: argmax baseline; "
                       "knn_lm: kNN-LM probability interpolation (L2 FAISS datastore); "
                       "rcd: RAD, retrieval-augmented contrastive decoding (cosine FAISS datastore); "
                       "cad: context-aware decoding, contrasts full vs. bare-question prompt; "
                       "instructive: instructive decoding, contrasts standard vs. adversarial prompt; "
                       "dola: divergence-of-layers decoding. "
                       "--- PROMPT-AUGMENTATION / ICL (inject retrieved info into prompt) --- "
                       "icl: standard few-shot prompt, no retrieval; "
                       "KAPING: injects top-k REBEL knowledge-graph triplets into few_shot prompt; "
                       "kNN-ICL: injects top-k similar training Q&A pairs into few_shot_icl prompt."
                   ))
    p.add_argument("--prompt_key",       default="zero_shot")
    p.add_argument("--noisy_prompt_key", default=None)
    p.add_argument("--eta",             type=float, default=0.3,
                   help="Contrastive strength for instructive/cad.")
    p.add_argument("--embed_model_name", default="all-MiniLM-L6-v2")

    # Configs sweep
    p.add_argument("--configs_json",     default=None,
                   help="JSON array of per-run config overrides. "
                        "Defaults to built-in sweep for the (method, eval_data) pair.")

    # Evaluation
    p.add_argument("--eval_metric",      default="factuality",
                   choices=["factuality", "truth_only", "halu_rate", "faith",
                             "open", "precisewiki", "hhem"])
    p.add_argument("--evaluation_type",  default="cohere",
                   choices=["cohere", "gemini"])
    p.add_argument("--save_results",     action="store_true")
    p.add_argument("--run_only",         action="store_true",
                   help="Skip Cohere/Gemini evaluation; only run inference. "
                        "Saves responses + a _timing.json with latency stats.")
    p.add_argument("--save_decoding_stats", action="store_true",
                   help="Save per-token entropy/prob stats to a _stats.pkl alongside "
                        "the response JSON. Used for diversity/entropy analysis.")

    # Misc
    p.add_argument("--batch_size",       type=int, default=16)
    p.add_argument("--seed",             type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = get_args()

    # Resolve model path
    args.model_path = get_model_path(args.base_model)

    # Default train_data if not provided (use eval_data)
    if not args.train_data:
        args.train_data = args.eval_data

    # Set up LLM eval client
    if not args.run_only:
        from utils import api_key
        if args.evaluation_type == "cohere":
            import cohere
            args.cohere_client = cohere.ClientV2(api_key=api_key.cohere_api_key)
        elif args.evaluation_type == "gemini":
            from google import genai
            if not api_key.gemini_api_key:
                raise ValueError(
                    "GEMINI_API_KEY not set. Add it to local/keys.env or export it as an environment variable."
                )
            args.gemini_client = genai.Client(api_key=api_key.gemini_api_key)

    # Build config sweep
    if args.configs_json:
        all_configs = json.loads(args.configs_json)
    else:
        all_configs = _get_default_configs(args.decoding_method, args.eval_data)

    # Methods that always run a single config (no sweep)
    if args.decoding_method in {"greedy", "dola", "icl", "KAPING", "kNN-ICL"}:
        all_configs = [{}]

    # Run sweep
    for cfg_extra in tqdm(all_configs, desc="Config sweep"):
        if args.decoding_method in {"knn_lm", "rcd"}:
            decoding_config = _build_rcd_cfg(args, cfg_extra)
        elif args.decoding_method == "instructive":
            decoding_config = _build_instructive_cfg(args, cfg_extra)
        elif args.decoding_method == "cad":
            decoding_config = _build_cad_cfg(args, cfg_extra)
        else:
            decoding_config = dict(cfg_extra)

        suffix = _config_to_str(cfg_extra) if cfg_extra else ""

        try:
            run_single(args, decoding_config=decoding_config, suffix=suffix)
        except Exception as exc:
            print(f"[ERROR] config={cfg_extra}: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
