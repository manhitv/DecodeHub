#!/bin/bash
# Prefetch the sentence embedders used by scripts/run_embed_sensitivity.sh.
# Run this on a node with internet (e.g. the login node) so the weights are in the
# HuggingFace cache before an offline compute job starts. Re-running is a no-op.
set -e
cd "$(dirname "$0")/.."

python - <<'PY'
from sentence_transformers import SentenceTransformer

EMBEDDERS = [
    "sentence-transformers/all-MiniLM-L6-v2",      # d=384 (default)
    "sentence-transformers/all-mpnet-base-v2",     # d=768
    "sentence-transformers/all-roberta-large-v1",  # d=1024
]
for name in EMBEDDERS:
    print(f"[prefetch] {name}")
    m = SentenceTransformer(name)
    print(f"          dim={m.get_sentence_embedding_dimension()}")
print("Done — embedders cached.")
PY
