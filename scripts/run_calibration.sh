#!/bin/bash
# Where RAD intervenes and its effect on calibration (Fig 5), on TruthfulQA /
# Qwen2.5-7B. Generates Greedy and RAD answers with per-token decoding stats and
# per-sample truthfulness labels, then draws the combined figure:
#   (a) selectivity  - fraction of tokens RAD changes vs. base next-token entropy
#   (b) reliability  - Greedy vs. RAD reliability diagram with ECE + AUROC
# Requires COHERE_API_KEY (truthfulness labels come from the standard judge).
set -e
cd "$(dirname "$0")/.."

export DECODEHUB_OUTPUT="${DECODEHUB_OUTPUT:-/weka/$USER/decodehub/output}"
RES=results/calibration
mkdir -p "$DECODEHUB_OUTPUT" "$RES"

MODEL=qwen2.5-7b
DATA=truthful_qa
NUM_TRAIN=100
N_SAMPLES=417
EMB=all-MiniLM-L6-v2
TAG=calib

python -m database.datastore --base_model "$MODEL" --train_data "$DATA" \
  --base_embed_model "$EMB" --num_train "$NUM_TRAIN" --chunk_size 8 --skip_if_exists

# batch_size 1 keeps the per-token stats clean (no post-EOS padding tokens).
echo "########## Greedy — generate + stats + per-sample labels ##########"
python run.py --base_model "$MODEL" --decoding_method greedy \
  --eval_data "$DATA" --max_sample_num "$N_SAMPLES" --batch_size 1 \
  --eval_metric truth_only --save_decoding_stats \
  --save_per_sample --per_sample_out "$RES/greedy_truth.json" --exp_tag "$TAG"

echo "########## RAD — generate + stats + per-sample labels ##########"
python run.py --base_model "$MODEL" --decoding_method rcd \
  --eval_data "$DATA" --train_data "$DATA" --num_train "$NUM_TRAIN" \
  --embed_model_name "$EMB" --max_sample_num "$N_SAMPLES" --batch_size 1 \
  --eval_metric truth_only --save_decoding_stats \
  --save_per_sample --per_sample_out "$RES/rad_truth.json" --exp_tag "$TAG" \
  --configs_json '[{"shaping_mode":"linear","alpha":0.5,"sim_threshold":0.7,"agg_mode":"weighted"}]'

GREEDY_STATS=$(ls -t "$DECODEHUB_OUTPUT"/E${DATA}_*_${MODEL}_greedy_test_${N_SAMPLES}_*exp_${TAG}_stats.pkl | head -1)
RAD_STATS=$(ls -t "$DECODEHUB_OUTPUT"/E${DATA}_*_${MODEL}_rcd_test_${N_SAMPLES}_*exp_${TAG}_stats.pkl | head -1)

# Combined calibration figure. The RAD stats file carries both the per-token
# confidence and the base-entropy / flip instrumentation, so it serves as both.
python analysis.py calib-combined \
  --greedy_stats "$GREEDY_STATS" --greedy_truth "$RES/greedy_truth.json" \
  --rad_stats    "$RAD_STATS"    --rad_truth    "$RES/rad_truth.json" \
  --flip_stats   "$RAD_STATS" \
  --out "$RES/calibration_combined.pdf" --csv_out "$RES/calibration_combined.csv"

echo "=== Done — see $RES/calibration_combined.pdf (+ .csv) ==="
