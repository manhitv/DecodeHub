#!/bin/bash
# Effect of the sentence embedder on RAD (Table 5): the same RAD run on
# TruthfulQA / Qwen2.5-7B with three embedders of increasing dimension.
#   MiniLM   all-MiniLM-L6-v2     d=384   (default)
#   MPNet    all-mpnet-base-v2    d=768
#   RoBERTa  all-roberta-large-v1 d=1024
# Requires COHERE_API_KEY. On compute nodes without internet, run
# scripts/prefetch_embedders.sh on the login node first.
set -e
cd "$(dirname "$0")/.."

# Large grounding-space .pt files -> /weka, not the home partition (HPC default).
export DECODEHUB_OUTPUT="${DECODEHUB_OUTPUT:-/weka/$USER/decodehub/output}"
# Isolate this experiment's eval records so the chart sees only these three runs.
export DECODEHUB_RESULTS=results/embed_sensitivity
mkdir -p "$DECODEHUB_OUTPUT" "$DECODEHUB_RESULTS"

MODEL=qwen2.5-7b
DATA=truthful_qa
NUM_TRAIN=100
N_SAMPLES=417
CFG='[{"shaping_mode":"linear","alpha":0.5,"sim_threshold":0.7,"agg_mode":"weighted"}]'

EMBEDDERS=(
  "all-MiniLM-L6-v2"
  "all-mpnet-base-v2"
  "all-roberta-large-v1"
)

for EMB in "${EMBEDDERS[@]}"; do
  echo "########## embedder=$EMB ##########"
  python -m database.datastore --base_model "$MODEL" --train_data "$DATA" \
    --base_embed_model "$EMB" --num_train "$NUM_TRAIN" --chunk_size 8 --skip_if_exists
  python run.py --base_model "$MODEL" --decoding_method rcd \
    --eval_data "$DATA" --train_data "$DATA" --num_train "$NUM_TRAIN" \
    --embed_model_name "$EMB" --eval_metric factuality \
    --max_sample_num "$N_SAMPLES" --configs_json "$CFG"
done

# Grouped-bar chart of %Truth / %Info / T*I per embedder.
python analysis.py embed-sensitivity \
  --result_file "$DECODEHUB_RESULTS/${DATA}_${MODEL}_rcd.jsonl" \
  --out "$DECODEHUB_RESULTS/embed_sensitivity.pdf" \
  --csv_out "$DECODEHUB_RESULTS/embed_sensitivity.csv"

echo "=== Done — see $DECODEHUB_RESULTS/embed_sensitivity.pdf (+ .csv) ==="
