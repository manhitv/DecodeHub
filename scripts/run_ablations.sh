#!/bin/bash
# RAD ablation studies (paper Section 4.3), on TruthfulQA, metric = factuality.
#   Block 1 - grounding space size N in {10, 50, 100, 200, 400}   (all four models)
#   Block 2 - chunk size M in {4, 8, 16, full}                    (Qwen2.5-7B)
#   Block 3 - similarity threshold tau in {0.1 .. 1.0}            (Qwen2.5-7B)
#   Block 4 - interpolation weight alpha in {0, 0.1, 0.2, 0.5, 1} (Qwen2.5-7B)
#   Block 5 - exact-match retrieval configuration                (Qwen2.5-7B)
#
# Paper -> code: RAD = rcd.  alpha=0 or tau=1 reproduces the Greedy baseline.
set -e
cd "$(dirname "$0")/.."
mkdir -p output results evaluation_results

DATA=truthful_qa
METRIC=factuality
N_SAMPLES=100

rad () {  # rad <model> <num_train> <configs_json>
  python run.py --base_model "$1" --decoding_method rcd \
    --eval_data "$DATA" --train_data "$DATA" --num_train "$2" \
    --embed_model_name all-MiniLM-L6-v2 \
    --eval_metric "$METRIC" --max_sample_num "$N_SAMPLES" \
    --configs_json "$3"
}

DEFAULT_CFG='{"shaping_mode":"linear","alpha":0.5,"sim_threshold":0.7,"agg_mode":"weighted"}'

echo "=== Block 1: grounding space size N in {10,50,100,200,400} (all models) ==="
for MODEL in qwen2.5-3b qwen2.5-7b mistral2-7b gemma2-9b; do
  for N in 10 50 100 200 400; do
    python -m database.datastore --base_model "$MODEL" --train_data "$DATA" \
      --num_train "$N" --chunk_size 8 --skip_if_exists
    rad "$MODEL" "$N" "[$DEFAULT_CFG]"
  done
done

# Remaining blocks use Qwen2.5-7B as the representative model.
MODEL=qwen2.5-7b
python -m database.datastore --base_model "$MODEL" --train_data "$DATA" --num_train 100 --chunk_size 8 --skip_if_exists

echo "=== Block 2: chunk size M in {4,8,16,full} ==="
for M in 4 8 16; do
  python -m database.datastore --base_model "$MODEL" --train_data "$DATA" --num_train 100 --chunk_size "$M" --skip_if_exists
  rad "$MODEL" 100 "[{\"shaping_mode\":\"linear\",\"alpha\":0.5,\"sim_threshold\":0.7,\"agg_mode\":\"weighted\",\"chunk_size\":$M}]"
done
# 'full' context chunks (chunk_size 0 builds the full-context datastore)
python -m database.datastore --base_model "$MODEL" --train_data "$DATA" --num_train 100 --chunk_size 0 --skip_if_exists
rad "$MODEL" 100 '[{"shaping_mode":"linear","alpha":0.5,"sim_threshold":0.7,"agg_mode":"weighted","chunk_size":0}]'

echo "=== Block 3: similarity threshold tau in {0.1..1.0} ==="
rad "$MODEL" 100 '[
  {"shaping_mode":"linear","alpha":0.5,"agg_mode":"weighted","sim_threshold":0.1},
  {"shaping_mode":"linear","alpha":0.5,"agg_mode":"weighted","sim_threshold":0.3},
  {"shaping_mode":"linear","alpha":0.5,"agg_mode":"weighted","sim_threshold":0.5},
  {"shaping_mode":"linear","alpha":0.5,"agg_mode":"weighted","sim_threshold":0.7},
  {"shaping_mode":"linear","alpha":0.5,"agg_mode":"weighted","sim_threshold":0.8},
  {"shaping_mode":"linear","alpha":0.5,"agg_mode":"weighted","sim_threshold":0.9},
  {"shaping_mode":"linear","alpha":0.5,"agg_mode":"weighted","sim_threshold":1.0}
]'

echo "=== Block 4: interpolation weight alpha in {0,0.1,0.2,0.5,1} ==="
rad "$MODEL" 100 '[
  {"shaping_mode":"linear","alpha":0.0,"sim_threshold":0.7,"agg_mode":"weighted"},
  {"shaping_mode":"linear","alpha":0.1,"sim_threshold":0.7,"agg_mode":"weighted"},
  {"shaping_mode":"linear","alpha":0.2,"sim_threshold":0.7,"agg_mode":"weighted"},
  {"shaping_mode":"linear","alpha":0.5,"sim_threshold":0.7,"agg_mode":"weighted"},
  {"shaping_mode":"linear","alpha":1.0,"sim_threshold":0.7,"agg_mode":"weighted"}
]'

echo "=== Block 5: exact-match retrieval configuration ==="
rad "$MODEL" 100 '[
  {"shaping_mode":"linear","alpha":0.5,"sim_threshold":0.7,"agg_mode":"weighted","exact_match":""},
  {"shaping_mode":"linear","alpha":0.5,"sim_threshold":0.7,"agg_mode":"weighted","exact_match":"ignore"}
]'

echo "=== Ablations complete — see evaluation_results/ ==="
