#!/bin/bash
# Efficiency analysis (paper Section 4.3 + Appendix A.4):
#   - grounding-space construction time per dataset
#   - RAD retrieval latency per 8-token context (cosine FAISS)
# Produces the numbers behind the latency figure and the grounding-space statistics.
set -e
cd "$(dirname "$0")/.."
mkdir -p output results evaluation_results

MODEL=qwen2.5-7b
DATASETS="truthful_qa wiki alpaca halu_dia halu_sum"

echo "=== Grounding-space construction time + size per dataset ==="
# Building the chunk-8 datastore prints construction time and |C|; results cached.
for data in $DATASETS; do
  echo "--- $data ---"
  python -m database.datastore --base_model "$MODEL" --train_data "$data" \
    --num_train 100 --chunk_size 8 --skip_if_exists
done

echo "=== RAD retrieval latency per 8-token context (cosine FAISS) ==="
# Omit --use_l2 so the benchmark uses cosine similarity (RAD), not L2 (kNN-LM).
python analysis.py knn-latency \
  --base_model "$MODEL" \
  --datasets $DATASETS \
  --chunk_mode chunk_8 \
  --result_file results/rad_latency.csv

echo "=== Analysis complete — see results/rad_latency.csv ==="
