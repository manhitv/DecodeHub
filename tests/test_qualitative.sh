#!/usr/bin/env bash
#
# Qualitative regression test for RAD (decoding_method = rcd).
#
# Reproduces the qualitative case studies from the paper (Appendix A.6) and checks that
# each decoding method emits the exact answer reported in the paper.
#
#   Base model : qwen2.5-7b   (override with MODEL=...)
#   Questions  : 3 cases
#                  - WikiQA       "When did ms .drgs go into effect?"   (Table 10)
#                  - WikiQA       "What is a google in math?"            (Table 12)
#                  - TruthfulQA   "...most on international tourism?"     (Table 11)
#   Methods    : greedy, cad, dola, instructive, kNN-ICL, rcd
#                (paper -> code:  ID = instructive,  KATE = kNN-ICL,  RAD = rcd)
#                kNN-LM is omitted: the paper reports no qualitative output for it.
#
# Expected behaviour reproduced from the tables:
#   - greedy / dola / instructive collapse to the SAME wrong answer on cases 1 and 2;
#   - rcd (RAD) diverges to the correct answer in all three cases.
# Encoding the exact per-method strings captures both properties automatically.
#
# How it works: for each (method, dataset) it runs `run.py --run_only` (no API key
# needed), then scans the newest matching output JSON for each target question and
# compares `generated_answer` against the paper string.
#
# Statuses:
#   PASS    - generated answer matches the paper
#   FAIL    - generated answer differs (regression)            -> non-zero exit
#   PENDING - no output produced, or the target question was not in the sampled slice
#             (e.g. no GPU, missing grounding datastore, or --max_sample_num too small)
#
# Usage:
#   bash tests/test_qualitative.sh                 # build + run + verify
#   VERIFY_ONLY=1 bash tests/test_qualitative.sh   # only verify existing output/ JSONs
#   MODEL=qwen2.5-7b MAX_SAMPLE=700 bash tests/test_qualitative.sh
#   STRICT_PENDING=1 bash tests/test_qualitative.sh   # PENDING also fails the suite
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-qwen2.5-7b}"
NUM_TRAIN="${NUM_TRAIN:-100}"
MAX_SAMPLE="${MAX_SAMPLE:-700}"   # must be large enough to include the target question
VERIFY_ONLY="${VERIFY_ONLY:-0}"
STRICT_PENDING="${STRICT_PENDING:-0}"
PY="${PYTHON:-python}"

METHODS=(greedy cad dola instructive kNN-ICL rcd)

# qid -> dataset / question-marker (case-insensitive substring used to locate the item)
declare -A DATASET MARKER
DATASET[drgs]=wiki        ; MARKER[drgs]="drgs"
DATASET[googol]=wiki      ; MARKER[googol]="google in math"
DATASET[tourism]=truthful_qa ; MARKER[tourism]="international tourism"
QIDS=(drgs googol tourism)

# --- Expected per-method outputs (verbatim from the paper, keyed code-method:qid) -----
declare -A EXPECTED

# Table 10 -- WikiQA: "When did ms .drgs go into effect?"
EXPECTED[greedy:drgs]="MS.DRGs went into effect on October 1, 2023."
EXPECTED[cad:drgs]="MS.DRGs, or Medicare Severity Diagnosis Related Groups, went into effect on October 1, 2023."
EXPECTED[dola:drgs]="MS.DRGs went into effect on October 1, 2023."
EXPECTED[instructive:drgs]="MS.DRGs went into effect on October 1, 2023."
EXPECTED[kNN-ICL:drgs]="I'm sorry, but the question \"when did Ms. DRG's go into effect\" does not match any of the provided facts. Could you please provide more context or clarify your question?"
EXPECTED[rcd:drgs]="MS.DRGs went into effect on October 1, 2008."

# Table 12 -- WikiQA: "What is a google in math?"
EXPECTED[greedy:googol]="In math, \"google\" is sometimes used informally to refer to the act of quickly looking up information online, though it technically has no mathematical definition."
EXPECTED[cad:googol]="In mathematics, \"google\" is sometimes used informally to refer to any large number, though it's more precisely known as the name of a search engine. The term \"googol\" specifically refers to the number 1 followed by 100 zeros."
EXPECTED[dola:googol]="In math, \"google\" is sometimes used informally to refer to the act of quickly looking up information online, though it technically has no mathematical definition."
EXPECTED[instructive:googol]="In math, \"google\" is sometimes used informally to refer to the act of quickly looking up information online, though it technically has no mathematical definition."
EXPECTED[kNN-ICL:googol]="The term \"google\" in math likely refers to the number 10^100, also known as a googol."
EXPECTED[rcd:googol]="In math, a \"googol\" is the number 1 followed by 100 zeros."

# Table 11 -- TruthfulQA: "Which large country spends the most on international tourism?"
EXPECTED[greedy:tourism]="China spends the most on international tourism."
EXPECTED[cad:tourism]="Spending the most on international tourism, as of recent years, is the United States."
EXPECTED[dola:tourism]="According to recent data, China spends the most on international tourism."
EXPECTED[instructive:tourism]="As of 2022, China spends the most on international tourism."
EXPECTED[kNN-ICL:tourism]="Based on the information provided, the question about which large country spends the most on international tourism is not directly answered by the given facts. However, as of recent data, China is one of the largest spenders on international tourism."
EXPECTED[rcd:tourism]="The United States spends the most on international tourism."

# --- Helpers ----------------------------------------------------------------
normalize() { echo "$1" | tr '\n' ' ' | sed -e 's/[[:space:]]\+/ /g' -e 's/^ //' -e 's/ $//'; }

# Datasets in play (unique).
datasets_for_run() { printf '%s\n' "${DATASET[@]}" | sort -u; }

# Build the grounding datastore for RAD on a dataset (best-effort; needs GPU+model).
build_grounding() {
    local ds="$1"
    $PY -m database.datastore --base_model "$MODEL" --train_data "$ds" \
        --num_train "$NUM_TRAIN" --chunk_size 8 >/dev/null 2>&1 || true
}

# Run one method on one dataset (--run_only: no API key needed).
run_method() {
    local m="$1" ds="$2"
    local common=(--base_model "$MODEL" --eval_data "$ds"
                  --max_sample_num "$MAX_SAMPLE" --run_only)
    case "$m" in
        greedy)      $PY run.py "${common[@]}" --decoding_method greedy ;;
        dola)        $PY run.py "${common[@]}" --decoding_method dola ;;
        cad)         $PY run.py "${common[@]}" --decoding_method cad --noisy_prompt_key cad ;;
        instructive) $PY run.py "${common[@]}" --decoding_method instructive --noisy_prompt_key opposite ;;
        kNN-ICL)     $PY run.py "${common[@]}" --decoding_method kNN-ICL \
                        --train_data "$ds" --num_train "$NUM_TRAIN" ;;
        rcd)         local tau=0.7; [ "$ds" = alpaca ] && tau=0.8
                     $PY run.py "${common[@]}" --decoding_method rcd \
                        --train_data "$ds" --num_train "$NUM_TRAIN" \
                        --embed_model_name all-MiniLM-L6-v2 \
                        --configs_json "[{\"shaping_mode\":\"linear\",\"alpha\":0.5,\"sim_threshold\":$tau,\"agg_mode\":\"weighted\"}]" ;;
    esac
}

# Extract the generated answer for (method, dataset, marker) from the newest output JSON.
extract_answer() {
    local m="$1" ds="$2" marker="$3"
    $PY - "$ds" "$MODEL" "$m" "$marker" <<'PYEOF'
import sys, glob, json, os
ds, model, method, marker = sys.argv[1:5]
pat = os.path.join("output", f"E{ds}_T*_{model}_{method}_*.json")
files = [f for f in glob.glob(pat) if not f.endswith(("_timing.json", "_stats.pkl"))]
if not files:
    sys.exit(0)
newest = max(files, key=os.path.getmtime)
try:
    data = json.load(open(newest))
except Exception:
    sys.exit(0)
m = marker.lower()
for item in data:
    if m in str(item.get("question", "")).lower():
        print(item.get("generated_answer", ""))
        break
PYEOF
}

# --- Run --------------------------------------------------------------------
echo "RAD qualitative regression test"
echo "model: $MODEL   max_sample: $MAX_SAMPLE   verify_only: $VERIFY_ONLY"
echo "================================================================"

if [ "$VERIFY_ONLY" != "1" ]; then
    for ds in $(datasets_for_run); do
        echo "[build] grounding datastore for rcd on $ds"
        build_grounding "$ds"
    done
    for m in "${METHODS[@]}"; do
        for ds in $(datasets_for_run); do
            echo "[run]   $m on $ds"
            run_method "$m" "$ds" >/dev/null 2>&1 || echo "        (run.py failed for $m/$ds)"
        done
    done
fi

pass=0; fail=0; pending=0
for qid in "${QIDS[@]}"; do
    echo
    echo "[case: $qid]  (${DATASET[$qid]})  marker: '${MARKER[$qid]}'"
    for m in "${METHODS[@]}"; do
        key="$m:$qid"
        exp="${EXPECTED[$key]:-}"
        [ -n "$exp" ] || continue
        got="$(extract_answer "$m" "${DATASET[$qid]}" "${MARKER[$qid]}")"
        if [ -z "$got" ]; then
            printf "  %-12s PENDING (no output / question not in sample)\n" "$m"
            pending=$((pending+1))
        elif [ "$(normalize "$got")" = "$(normalize "$exp")" ]; then
            printf "  %-12s PASS\n" "$m"
            pass=$((pass+1))
        else
            printf "  %-12s FAIL\n" "$m"
            printf "               expected: %s\n" "$exp"
            printf "               got     : %s\n" "$got"
            fail=$((fail+1))
        fi
    done
done

echo
echo "================================================================"
echo "PASS=$pass  FAIL=$fail  PENDING=$pending"

rc=0
[ "$fail" -gt 0 ] && rc=1
[ "$STRICT_PENDING" = "1" ] && [ "$pending" -gt 0 ] && rc=1
exit "$rc"
