#!/bin/bash
# Main results (Tables 1-2): RAD vs. six baselines across the open-ended benchmarks.
# Methods : greedy, cad, dola, instructive (ID), kNN-ICL (KATE), knn_lm, rcd (RAD)
# Models  : qwen2.5-3b, qwen2.5-7b, mistral2-7b, gemma2-9b
# Requires COHERE_API_KEY for evaluation (use --run_only to skip).
#
# Paper -> code method names:  ID = instructive,  KATE = kNN-ICL,  RAD = rcd
set -e
cd "$(dirname "$0")/.."
mkdir -p output results evaluation_results

MODELS=("qwen2.5-3b" "qwen2.5-7b" "mistral2-7b" "gemma2-9b")
DATASETS=("truthful_qa" "wiki" "alpaca" "halu_dia" "halu_sum")
NUM_TRAIN=100
N_SAMPLES=100

# Evaluation metric per dataset.
get_metric () {
  case "$1" in
    halu_dia|halu_sum) echo halu_rate  ;;
    *)                 echo factuality ;;  # truthful_qa, wiki, alpaca
  esac
}

# RAD similarity threshold: tau=0.8 for Alpaca (diverse topics), 0.7 otherwise.
get_tau () {
  case "$1" in
    alpaca) echo 0.8 ;;
    *)      echo 0.7 ;;
  esac
}

for model in "${MODELS[@]}"; do
  for data in "${DATASETS[@]}"; do
    METRIC=$(get_metric "$data")
    TAU=$(get_tau "$data")

    echo "########## $model / $data (metric=$METRIC) ##########"

    # --- Precompute grounding structures (chunk-8 context + next-token logits) ---
    echo "--- [precompute] grounding space (RAD) + kNN-LM datastore ---"
    python -m database.datastore --base_model "$model" --train_data "$data" \
      --num_train "$NUM_TRAIN" --chunk_size 8 --skip_if_exists
    echo "--- [precompute] KAPING REBEL triplets (for kNN-ICL / KAPING) ---"
    python -m utils.baseline     --train_data "$data" --num_train "$NUM_TRAIN"

    # --- Baselines ----------------------------------------------------------
    python run.py --base_model "$model" --decoding_method greedy \
      --eval_data "$data" --eval_metric "$METRIC" --max_sample_num "$N_SAMPLES"
    python run.py --base_model "$model" --decoding_method cad \
      --eval_data "$data" --eval_metric "$METRIC" --max_sample_num "$N_SAMPLES" --noisy_prompt_key cad
    python run.py --base_model "$model" --decoding_method dola \
      --eval_data "$data" --eval_metric "$METRIC" --max_sample_num "$N_SAMPLES"
    python run.py --base_model "$model" --decoding_method instructive \
      --eval_data "$data" --eval_metric "$METRIC" --max_sample_num "$N_SAMPLES" --noisy_prompt_key opposite
    python run.py --base_model "$model" --decoding_method kNN-ICL \
      --eval_data "$data" --train_data "$data" --num_train "$NUM_TRAIN" \
      --eval_metric "$METRIC" --max_sample_num "$N_SAMPLES"
    python run.py --base_model "$model" --decoding_method knn_lm \
      --eval_data "$data" --train_data "$data" --num_train "$NUM_TRAIN" \
      --eval_metric "$METRIC" --max_sample_num "$N_SAMPLES"

    # --- RAD (Ours) ---------------------------------------------------------
    python run.py --base_model "$model" --decoding_method rcd \
      --eval_data "$data" --train_data "$data" --num_train "$NUM_TRAIN" \
      --embed_model_name all-MiniLM-L6-v2 \
      --eval_metric "$METRIC" --max_sample_num "$N_SAMPLES" \
      --configs_json "[{\"shaping_mode\":\"linear\",\"alpha\":0.5,\"sim_threshold\":$TAU,\"agg_mode\":\"weighted\"}]"
  done
done

echo "=== Main table complete — see evaluation_results/ ==="
