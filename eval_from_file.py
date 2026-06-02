#!/usr/bin/env python3
"""Evaluate a pre-generated output JSON without re-running inference.

Reads an existing output/*.json file, runs the Cohere evaluator, and saves
results to evaluation_results/ in the same format as run.py.

Usage:
    python eval_from_file.py \
        --file       output/Ebio_Tbio_N100_qwen2.5-3b_knn_lm_test_100.json \
        --eval_data  bio \
        --base_model qwen2.5-3b \
        --decoding_method knn_lm \
        --eval_metric factuality \
        --train_data bio \
        --max_sample_num 100 \
        [--decoding_config_json '{"alpha":1}']
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from utils import api_key
from utils.config import eval_result_dir
from utils.evaluation import eval_single_file


def main():
    p = argparse.ArgumentParser(description="Eval-only wrapper for pre-generated output files.")
    p.add_argument("--file",               required=True, help="Path to existing output JSON.")
    p.add_argument("--eval_data",          required=True)
    p.add_argument("--base_model",         required=True)
    p.add_argument("--decoding_method",    required=True)
    p.add_argument("--eval_metric",        default="factuality",
                   choices=["factuality", "halu_rate", "precisewiki", "faith", "open"])
    p.add_argument("--evaluation_type",    default="cohere", choices=["cohere", "gemini"])
    p.add_argument("--train_data",         default="")
    p.add_argument("--max_sample_num",     type=int, default=100)
    p.add_argument("--data_split",         default="test")
    p.add_argument("--max_new_tokens",     type=int, default=256)
    p.add_argument("--seed",               type=int, default=42)
    p.add_argument("--decoding_config_json", default="{}",
                   help="JSON string of decoding config (for result metadata).")
    p.add_argument("--save_per_sample", action="store_true",
                   help="Save per-sample labels to evaluation_results/per_sample/<name>.json. "
                        "For factuality: {question_index, is_correct} (used by the "
                        "calibration/ECE analysis); for precisewiki: hallucination flags.")
    p.add_argument("--per_sample_out", default=None,
                   help="Explicit path for the per-sample JSON (overrides the default "
                        "evaluation_results/per_sample/ location).")
    args = p.parse_args()

    if not args.train_data:
        args.train_data = args.eval_data

    args.eval_data_path = args.file
    args.decoding_config = json.loads(args.decoding_config_json)
    # Defaults expected by the evaluator but not exposed as CLI flags here.
    args.save_results = False
    args.exp_tag = ""

    # Resolve per-sample output path (honour an explicit --per_sample_out).
    if args.save_per_sample and not args.per_sample_out:
        per_sample_dir = os.path.join(eval_result_dir, "per_sample")
        per_sample_fn = (
            f"{args.decoding_method}__{args.eval_data}__{args.base_model}"
            f"__{args.data_split}__{args.max_sample_num}__per_sample.json"
        )
        args.per_sample_out = os.path.join(per_sample_dir, per_sample_fn)
    elif not args.save_per_sample:
        args.per_sample_out = None

    # Evaluation back-end.
    if args.evaluation_type == "cohere":
        import cohere
        args.cohere_client = cohere.ClientV2(api_key=api_key.cohere_api_key)
    elif args.evaluation_type == "gemini":
        from google import genai
        args.gemini_client = genai.Client(api_key=api_key.gemini_api_key)

    print(f"[EvalFromFile] {args.file}")
    result = eval_single_file(args.file, args=args)

    # Save to eval_results dir (identical format to run.py _save_eval_result)
    os.makedirs(eval_result_dir, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")
    fn = (
        f"{args.decoding_method}__{args.eval_data}__{args.base_model}"
        f"__{args.data_split}__{args.max_sample_num}__{ts}.json"
    )
    path = os.path.join(eval_result_dir, fn)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Eval saved] {path}")


if __name__ == "__main__":
    main()
