#!/bin/bash
# Quick end-to-end validation of the RAD pipeline (Qwen2.5-3B, WikiQA).
# Uses --run_only so it needs no API key and finishes in a few minutes.
set -e
cd "$(dirname "$0")/.."
mkdir -p output results evaluation_results

MODEL=qwen2.5-3b
DATA=wiki

echo "=== [1/3] Greedy baseline ==="
python run.py --base_model "$MODEL" --decoding_method greedy \
  --eval_data "$DATA" --max_sample_num 10 --run_only

echo "=== [2/3] Build grounding space (chunk-8 context + next-token logits) ==="
python -m database.datastore --base_model "$MODEL" --train_data "$DATA" \
  --num_train 50 --chunk_size 8 --skip_if_exists

echo "=== [3/3] RAD inference ==="
python run.py --base_model "$MODEL" --decoding_method rcd \
  --eval_data "$DATA" --train_data "$DATA" --num_train 50 \
  --embed_model_name all-MiniLM-L6-v2 \
  --configs_json '[{"shaping_mode":"linear","alpha":0.5,"sim_threshold":0.7,"agg_mode":"weighted"}]' \
  --max_sample_num 10 --run_only

echo "=== Validation complete — see output/ for generations and timing ==="
