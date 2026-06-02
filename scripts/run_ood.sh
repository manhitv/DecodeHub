#!/bin/bash
# Out-of-distribution / cross-dataset evaluation (Table 3): the grounding space is
# built on one dataset and used to decode the other (WikiQA -> TruthfulQA and vice
# versa), with only 100 training instances.
# Methods : dola, kNN-ICL (KATE), knn_lm, rcd (RAD).  RAD = rcd.
# Requires COHERE_API_KEY for evaluation (use --run_only to skip).
set -e
cd "$(dirname "$0")/.."
mkdir -p output results evaluation_results

MODELS=("qwen2.5-3b" "qwen2.5-7b" "mistral2-7b" "gemma2-9b")
NUM_TRAIN=100
N_SAMPLES=417          # TruthfulQA test split; WikiQA loader caps at its full split
METRIC=factuality

# Cross-dataset pairs: "<train>:<eval>"
PAIRS=("wiki:truthful_qa" "truthful_qa:wiki")

for model in "${MODELS[@]}"; do
  for pair in "${PAIRS[@]}"; do
    TRAIN="${pair%%:*}"; EVAL="${pair##*:}"
    echo "########## $model  train=$TRAIN -> eval=$EVAL ##########"

    python -m database.datastore --base_model "$model" --train_data "$TRAIN" \
      --num_train "$NUM_TRAIN" --chunk_size 8 --skip_if_exists

    # DoLa needs no grounding (strongest data-free baseline, shown for reference).
    python run.py --base_model "$model" --decoding_method dola \
      --eval_data "$EVAL" --eval_metric "$METRIC" --max_sample_num "$N_SAMPLES"
    python run.py --base_model "$model" --decoding_method kNN-ICL \
      --eval_data "$EVAL" --train_data "$TRAIN" --num_train "$NUM_TRAIN" \
      --eval_metric "$METRIC" --max_sample_num "$N_SAMPLES"
    python run.py --base_model "$model" --decoding_method knn_lm \
      --eval_data "$EVAL" --train_data "$TRAIN" --num_train "$NUM_TRAIN" \
      --eval_metric "$METRIC" --max_sample_num "$N_SAMPLES"
    python run.py --base_model "$model" --decoding_method rcd \
      --eval_data "$EVAL" --train_data "$TRAIN" --num_train "$NUM_TRAIN" \
      --embed_model_name all-MiniLM-L6-v2 \
      --eval_metric "$METRIC" --max_sample_num "$N_SAMPLES" \
      --configs_json '[{"shaping_mode":"linear","alpha":0.5,"sim_threshold":0.7,"agg_mode":"weighted"}]'
  done
done

echo "=== OOD evaluation complete — see evaluation_results/ ==="
