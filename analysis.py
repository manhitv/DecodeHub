"""
analysis.py — Consolidated latency benchmarks and analysis utilities.

Sub-commands
------------
knn-latency       Benchmark RAD retrieval datastore retrieval time (Fig. 6).
plot-ablation     Bar chart for the grounding-size ablation (Fig. 3).
embed-sensitivity Grouped-bar chart of scores per sentence embedder (Table 5).
entropy-flip      Token-flip rate vs. base entropy, RAD selectivity (Fig. 5a).
ece               Reliability diagram + ECE, Greedy vs. RAD (Fig. 5b).
calib-combined    Single figure combining selectivity + reliability (Fig. 5).

Usage examples
--------------
python analysis.py knn-latency \
    --base_model qwen2.5-7b --datasets truthful_qa wiki

python analysis.py plot-ablation \
    --result_dir results/ --method rcd --eval_data truthful_qa \
    --out results/ablation.png
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def compute_stats(x: list[float]) -> dict:
    arr = np.array(x)
    return {
        "mean":   float(arr.mean()),
        "std":    float(arr.std()),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
        "median": float(np.median(arr)),
    }


# ---------------------------------------------------------------------------
# RAD / kNN-LM datastore retrieval latency
# ---------------------------------------------------------------------------

def benchmark_knn_latency(
    base_model:       str,
    embed_model_name: str,
    datasets:         list[str],
    chunk_mode:       str  = "chunk_8",
    num_runs:         int  = 100_000,
    result_file:      str  = "results/knn_latency.csv",
    use_cosine:       bool = True,
) -> None:
    """Benchmark per-step RAD retrieval latency. Writes a CSV."""
    import faiss
    from sentence_transformers import SentenceTransformer
    from utils.config import output_dir

    device       = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model  = SentenceTransformer(f"sentence-transformers/{embed_model_name}", device=device)

    _ensure_dir(result_file)
    with open(result_file, "w") as f:
        f.write("n_runs,dataset,chunk_mode,time_seconds,avg_retrieval_time\n")

    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        data_path = os.path.join(output_dir, f"{dataset}_{base_model}_{embed_model_name}_context_{chunk_mode}.pt")
        if not os.path.exists(data_path):
            print(f"  [Skip] datastore not found: {data_path}")
            continue

        data = torch.load(data_path, weights_only=True)
        all_embs = torch.stack([d["context_embedding"] for d in data]).float()
        if use_cosine:
            all_embs = F.normalize(all_embs, dim=-1)
        emb_np = all_embs.cpu().numpy().astype("float32")
        dim    = emb_np.shape[1]

        if torch.cuda.is_available():
            res       = faiss.StandardGpuResources()
            cpu_index = faiss.IndexFlatIP(dim) if use_cosine else faiss.IndexFlatL2(dim)
            index     = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            index = faiss.IndexFlatIP(dim) if use_cosine else faiss.IndexFlatL2(dim)
        index.add(emb_np)

        vocab_size = data[0]["logits"].shape[0]
        chunk_n    = int(chunk_mode.split("_")[1]) if chunk_mode.startswith("chunk_") else 8

        def _random_context_text(n=chunk_n):
            return " ".join(str(random.randint(0, 9999)) for _ in range(n))

        sim_threshold = 0.7

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        for _ in range(num_runs):
            ctx_text = _random_context_text()
            ctx_emb  = embed_model.encode(ctx_text, convert_to_tensor=True).float()
            if use_cosine:
                ctx_emb = F.normalize(ctx_emb, dim=-1)
            q = ctx_emb.cpu().numpy().astype("float32").reshape(1, -1)
            k = min(2048, len(data))
            sims, idxs = index.search(q, k)

            _ = _compute_rcd_logits(
                sims[0].tolist(), idxs[0].tolist(), data,
                vocab_size, sim_threshold, device
            )

        elapsed  = time.time() - t0
        avg_time = elapsed / num_runs
        print(f"  Avg per call: {avg_time:.2e}s  (total={elapsed:.1f}s)")

        with open(result_file, "a") as f:
            f.write(f"{num_runs},{dataset},{chunk_mode},{elapsed:.6f},{avg_time:.8f}\n")

    print(f"\nResults saved to {result_file}")


def _compute_rcd_logits(sims, idxs, data, vocab_size, sim_threshold, device):
    """Inline replica of RCD logit aggregation for latency measurement."""
    filtered = [(s, i) for s, i in zip(sims, idxs) if s > sim_threshold]
    if not filtered:
        return torch.zeros(vocab_size, device=device)
    weights = torch.tensor([s for s, _ in filtered], device=device)
    weights = weights / weights.sum()
    vecs = torch.stack([data[i]["logits"].to(dtype=torch.float32, device=device) for _, i in filtered])
    return (vecs * weights.unsqueeze(1)).sum(dim=0)


# ---------------------------------------------------------------------------
# 3. Training-size ablation bar chart
# ---------------------------------------------------------------------------

def plot_training_size_ablation(
    result_dir:   str,
    method:       str,
    eval_data:    str,
    metric_key:   str = "metric__truth_score",
    out_path:     str = "results/ablation.png",
) -> None:
    """Read eval JSONL files and plot a bar chart grouped by num_train."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Error] matplotlib not installed. Run: pip install matplotlib")
        return

    records = []
    for fname in sorted(Path(result_dir).glob("*.jsonl")):
        for line in fname.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("decoding_method") == method and r.get("eval_data") == eval_data:
                records.append(r)

    if not records:
        print(f"[Warning] No records found for method={method}, eval_data={eval_data}")
        return

    from collections import defaultdict
    by_num_train: dict[int, list[float]] = defaultdict(list)
    for r in records:
        # num_train is flattened onto the record as "<method>__num_train" by run.py.
        nt = r.get("num_train") or r.get(f"{method}__num_train") or r.get("sample_num", 0)
        v  = r.get("eval_results", {}).get(metric_key)
        if v is not None:
            by_num_train[nt].append(v)

    num_trains = sorted(by_num_train)
    means  = [np.mean(by_num_train[n]) for n in num_trains]
    stds   = [np.std(by_num_train[n])  for n in num_trains]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(n) for n in num_trains], means, yerr=stds, capsize=5, color="steelblue")
    ax.set_xlabel("Training samples")
    ax.set_ylabel(metric_key.replace("metric__", ""))
    ax.set_title(f"{method} — {eval_data} — training size ablation")
    ax.set_ylim(0, 1.05)

    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")



# ---------------------------------------------------------------------------
# Paper-style plotting helpers (serif font + light grid, no seaborn dependency)
# ---------------------------------------------------------------------------

# Marker / colour palette matching the paper figures (Figs 3-4).
_PAPER_COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#ff0000"]
_PAPER_MARKERS = ["o", "s", "^", "D", "v"]
# Soft red for the entropy-flip bars (lighter than the palette red, matching edge).
_FLIP_BAR_COLOR = "#ef8a82"


def _use_paper_style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"]        = "serif"
    plt.rcParams["mathtext.fontset"]   = "stix"
    plt.rcParams["axes.grid"]          = True
    plt.rcParams["grid.linestyle"]     = "--"
    plt.rcParams["grid.linewidth"]     = 0.6
    plt.rcParams["grid.alpha"]         = 0.6
    plt.rcParams["axes.axisbelow"]     = True


# Sentence-embedder display names + dimensionality (for the sensitivity figure).
_EMBED_DISPLAY: dict[str, tuple[str, int]] = {
    "all-MiniLM-L6-v2":     ("MiniLM",  384),
    "all-mpnet-base-v2":    ("MPNet",   768),
    "all-roberta-large-v1": ("RoBERTa", 1024),
}


def _embed_label(name: str) -> tuple[str, int]:
    tag = Path(name).name
    if tag in _EMBED_DISPLAY:
        return _EMBED_DISPLAY[tag]
    return tag, -1


# ---------------------------------------------------------------------------
# Embedding-model sensitivity
# ---------------------------------------------------------------------------

def plot_embed_sensitivity(
    result_file: str,
    out_path:    str = "results/embed_sensitivity.pdf",
    csv_out:     Optional[str] = None,
) -> None:
    """Grouped-bar chart of %Truth / %Info / T*I for each embedder.

    Reads a RAD eval JSONL (one line per run) produced by run.py, groups the
    records by their ``rcd__embed_model_name`` field, and keeps the most recent
    record per embedder. Shows how robust RAD is to the choice of sentence
    embedder used to build the grounding space.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Error] matplotlib not installed. Run: pip install matplotlib")
        return

    records = [
        json.loads(line)
        for line in Path(result_file).read_text().splitlines()
        if line.strip()
    ]
    if not records:
        print(f"[Warning] No records found in {result_file}")
        return

    # Keep the latest record per embedder, keyed by the filesystem-safe tag so
    # that "all-MiniLM-L6-v2" and "sentence-transformers/all-MiniLM-L6-v2"
    # collapse onto a single bar.
    by_embed: dict[str, dict] = {}
    for r in records:
        emb = r.get("rcd__embed_model_name") or r.get("embed_model_name") or "unknown"
        tag = Path(emb).name
        if tag not in by_embed or r.get("timestamp", 0) >= by_embed[tag].get("timestamp", 0):
            by_embed[tag] = r

    # Order by embedding dimension (small -> large) when known.
    embedders = sorted(by_embed, key=lambda e: (_embed_label(e)[1], e))
    labels, dims = zip(*(_embed_label(e) for e in embedders))
    xticklabels = [f"{lab}\n(d={d})" if d > 0 else lab for lab, d in zip(labels, dims)]

    def _pct(rec, key):  # eval scores are stored as fractions in [0, 1]
        return 100.0 * rec.get("eval_results", {}).get(key, float("nan"))

    truth = [_pct(by_embed[e], "metric__truth_score") for e in embedders]
    info  = [_pct(by_embed[e], "metric__info_score")  for e in embedders]
    ti    = [_pct(by_embed[e], "metric__t_times_i")   for e in embedders]

    if csv_out:
        _ensure_dir(csv_out)
        with open(csv_out, "w") as f:
            f.write("embedder,dim,truth,info,t_times_i\n")
            for e, lab, d, t, inf, p in zip(embedders, labels, dims, truth, info, ti):
                f.write(f"{Path(e).name},{d},{t:.2f},{inf:.2f},{p:.2f}\n")
        print(f"Saved → {csv_out}")

    _use_paper_style()
    fig, ax = plt.subplots(figsize=(6.2, 4))
    x      = np.arange(len(embedders))
    width  = 0.26
    series = [("%Truth", truth, _PAPER_COLORS[0]),
              ("%Info",  info,  _PAPER_COLORS[2]),
              ("T*I",    ti,    _PAPER_COLORS[4])]
    for k, (name, vals, color) in enumerate(series):
        bars = ax.bar(x + (k - 1) * width, vals, width, label=name,
                      color=color, edgecolor="black", linewidth=0.5)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, max(max(truth), max(info), max(ti)) * 1.18)
    ax.set_xlabel("Sentence embedder (embedding dimension $d$)")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False, fontsize=9)
    ax.grid(axis="x", visible=False)

    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_path}")

    # Comparison table (printed + appended to CSV).
    print("\nPer-embedder scores (%) on TruthfulQA / Qwen2.5-7B:")
    hdr = f"  {'Embedder':>10} {'dim':>5} {'%Truth':>8} {'%Info':>8} {'T*I':>8}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for e, lab, d, t, inf, p in zip(embedders, labels, dims, truth, info, ti):
        print(f"  {lab:>10} {d:>5} {t:>8.1f} {inf:>8.1f} {p:>8.1f}")


# ---------------------------------------------------------------------------
# Calibration / ECE: Greedy vs. RAD
# ---------------------------------------------------------------------------

def _load_confidences(stats_pkl: str) -> dict[int, float]:
    """Map question_index -> mean per-token top-probability (sequence confidence)."""
    import pickle
    with open(stats_pkl, "rb") as f:
        data = pickle.load(f)
    conf: dict[int, float] = {}
    for d in data:
        steps = d.get("step_stats", [])
        if not steps:
            continue
        conf[d["question_index"]] = float(np.mean([s["top_prob"] for s in steps]))
    return conf


def _load_correctness(truth_json: str) -> dict[int, int]:
    """Map question_index -> 1 if the generated answer was judged truthful."""
    data = json.load(open(truth_json))
    return {int(d["question_index"]): int(bool(d["is_correct"])) for d in data}


def _auroc(scores: list[float], correct: list[int]) -> float:
    """Area under ROC: probability a correct sample is ranked above a wrong one.

    Measures *resolution* — how well confidence separates truthful from
    untruthful answers (0.5 = chance). Robust to overconfidence/miscalibration.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(correct, dtype=int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum((p > neg).sum() + 0.5 * (p == neg).sum() for p in pos)
    return float(wins / (len(pos) * len(neg)))


def _brier(confidences: list[float], correct: list[int]) -> float:
    """Brier score = mean squared error between confidence and correctness."""
    return float(np.mean((np.asarray(confidences) - np.asarray(correct, dtype=float)) ** 2))


def compute_ece(
    confidences: list[float], correct: list[int], n_bins: int = 10
) -> tuple[float, list[dict]]:
    """Expected Calibration Error with equal-width confidence bins.

    ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|.
    Returns (ece, per-bin diagnostics).
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    n    = len(conf)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            bins.append({"lo": lo, "hi": hi, "count": 0, "acc": None, "conf": None})
            continue
        acc  = float(corr[mask].mean())
        cbin = float(conf[mask].mean())
        ece += (cnt / n) * abs(acc - cbin)
        bins.append({"lo": lo, "hi": hi, "count": cnt, "acc": acc, "conf": cbin})
    return ece, bins


def plot_calibration(
    greedy_stats: str,
    greedy_truth: str,
    rad_stats:    str,
    rad_truth:    str,
    n_bins:       int = 10,
    out_path:     str = "results/calibration_ece.pdf",
    csv_out:      Optional[str] = None,
) -> None:
    """Reliability diagram + ECE comparing Greedy (before) vs. RAD (after).

    Sequence confidence = mean per-token top-probability of the generated answer
    (from a --save_decoding_stats .pkl). Correctness = truthfulness judgment
    (from a --save_per_sample eval dump). Lower ECE = better calibrated.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Error] matplotlib not installed. Run: pip install matplotlib")
        return

    methods = {
        "Greedy": (_load_confidences(greedy_stats), _load_correctness(greedy_truth)),
        "RAD":    (_load_confidences(rad_stats),    _load_correctness(rad_truth)),
    }

    results: dict[str, dict] = {}
    for name, (conf_map, corr_map) in methods.items():
        idxs = sorted(set(conf_map) & set(corr_map))
        if not idxs:
            print(f"[Warning] No overlapping samples for {name}.")
            continue
        confs = [conf_map[i] for i in idxs]
        corrs = [corr_map[i] for i in idxs]
        ece, bins = compute_ece(confs, corrs, n_bins=n_bins)
        auroc = _auroc(confs, corrs)
        brier = _brier(confs, corrs)
        results[name] = {
            "ece": ece, "bins": bins, "n": len(idxs), "auroc": auroc, "brier": brier,
            "mean_conf": float(np.mean(confs)), "acc": float(np.mean(corrs)),
        }
        print(f"{name}: ECE={ece:.4f}  AUROC={auroc:.4f}  Brier={brier:.4f}  "
              f"n={len(idxs)}  mean_conf={np.mean(confs):.3f}  accuracy={np.mean(corrs):.3f}")

    if not results:
        print("[Error] Nothing to plot — check stats/truth file alignment.")
        return

    if csv_out:
        _ensure_dir(csv_out)
        with open(csv_out, "w") as f:
            f.write("# summary metrics\n")
            f.write("method,n,accuracy,mean_conf,ece,auroc,brier\n")
            for name, r in results.items():
                f.write(f"{name},{r['n']},{r['acc']:.4f},{r['mean_conf']:.4f},"
                        f"{r['ece']:.4f},{r['auroc']:.4f},{r['brier']:.4f}\n")
            f.write("\n# reliability bins\n")
            f.write("method,bin_lo,bin_hi,count,confidence,accuracy\n")
            for name, r in results.items():
                for b in r["bins"]:
                    if b["count"] == 0:
                        continue
                    f.write(f"{name},{b['lo']:.3f},{b['hi']:.3f},"
                            f"{b['count']},{b['conf']:.4f},{b['acc']:.4f}\n")
        print(f"Saved → {csv_out}")

    _use_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) Reliability diagram: empirical accuracy vs. confidence.
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Perfect calibration")
    style = {"Greedy": (_PAPER_COLORS[0], _PAPER_MARKERS[0]),
             "RAD":    (_PAPER_COLORS[4], _PAPER_MARKERS[1])}
    for name, r in results.items():
        color, marker = style.get(name, (_PAPER_COLORS[1], "o"))
        xs = [b["conf"] for b in r["bins"] if b["count"] > 0]
        ys = [b["acc"]  for b in r["bins"] if b["count"] > 0]
        ax.plot(xs, ys, marker=marker, color=color, lw=1.8,
                label=f"{name} (ECE={r['ece']:.3f}, AUROC={r['auroc']:.3f})")
    ax.set_xlabel("Confidence (mean token probability)")
    ax.set_ylabel("Empirical truthfulness")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("(a) Reliability diagram")
    ax.legend(loc="upper left", fontsize=9, frameon=True)

    # (b) Confidence distribution (shows the sharpening induced by aggregation).
    ax = axes[1]
    for name, r in results.items():
        color, _ = style.get(name, (_PAPER_COLORS[1], "o"))
        conf_map, corr_map = methods[name]
        idxs = sorted(set(conf_map) & set(corr_map))
        confs = [conf_map[i] for i in idxs]
        ax.hist(confs, bins=20, range=(0, 1), alpha=0.5, density=True,
                color=color, label=f"{name} (mean={r['mean_conf']:.2f})")
        ax.axvline(r["mean_conf"], color=color, linestyle=":", lw=1.5)
    ax.set_xlabel("Confidence (mean token probability)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.set_title("(b) Confidence distribution")
    ax.legend(loc="upper left", fontsize=9, frameon=True)

    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Selectivity: where RAD intervenes vs. base entropy
# ---------------------------------------------------------------------------

def plot_entropy_flip(
    rad_stats: str,
    n_bins:    int = 4,
    out_path:  str = "results/entropy_flip.pdf",
    csv_out:   Optional[str] = None,
) -> None:
    """Fraction of decoding steps where RAD changes the token, binned by BASE
    entropy. Shows RAD acts as a targeted corrector in uncertain regions:
    near-zero flips at low base entropy, rising sharply at high entropy.

    Requires a RAD _stats.pkl produced after the per-step instrumentation
    (fields: base_entropy, flipped). Needs no truthfulness labels.
    """
    import pickle
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Error] matplotlib not installed."); return

    with open(rad_stats, "rb") as f:
        data = pickle.load(f)

    base_ent, flipped, gaps = [], [], []
    for d in data:
        for s in d.get("step_stats", []):
            if "base_entropy" not in s or "flipped" not in s:
                continue
            base_ent.append(s["base_entropy"])
            flipped.append(s["flipped"])
            gaps.append(s.get("base_gap", float("nan")))
    if not base_ent:
        print("[Error] No base_entropy/flipped fields — regenerate RAD stats with "
              "the instrumented rcd_generation (--save_decoding_stats).")
        return

    base_ent = np.asarray(base_ent); flipped = np.asarray(flipped, dtype=float)
    gaps = np.asarray(gaps)
    n = len(base_ent)

    # Equal-frequency (quartile) bins by base entropy.
    edges = np.quantile(base_ent, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    rows = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (base_ent >= lo) & (base_ent < hi)
        cnt = int(m.sum())
        rows.append({
            "bin": b + 1, "lo": float(lo), "hi": float(hi), "count": cnt,
            "flip_rate": float(flipped[m].mean()) if cnt else float("nan"),
            "mean_gap":  float(np.nanmean(gaps[m])) if cnt else float("nan"),
        })

    print(f"Selectivity (n={n} steps, overall flip rate={flipped.mean():.3f}):")
    print(f"  {'quartile':>8} {'base-entropy range':>22} {'steps':>7} {'flip%':>8} {'mean top1-2 gap':>16}")
    qnames = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"] if n_bins == 4 else [f"Q{r['bin']}" for r in rows]
    for q, r in zip(qnames, rows):
        print(f"  {q:>8} [{r['lo']:5.2f}, {r['hi']:5.2f}] {r['count']:>7} "
              f"{100*r['flip_rate']:>7.1f}% {r['mean_gap']:>16.3f}")

    if csv_out:
        _ensure_dir(csv_out)
        with open(csv_out, "w") as f:
            f.write("quartile,base_entropy_lo,base_entropy_hi,steps,flip_rate,mean_top1_top2_gap\n")
            for q, r in zip(qnames, rows):
                f.write(f"{q},{r['lo']:.4f},{r['hi']:.4f},{r['count']},"
                        f"{r['flip_rate']:.4f},{r['mean_gap']:.4f}\n")
        print(f"Saved → {csv_out}")

    _use_paper_style()
    fig, ax = plt.subplots(figsize=(5.4, 4))
    xs = np.arange(len(rows))
    bars = ax.bar(xs, [100 * r["flip_rate"] for r in rows], width=0.6,
                  color=_FLIP_BAR_COLOR, edgecolor=_FLIP_BAR_COLOR, linewidth=0.5)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=9, padding=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{q}\n[{r['lo']:.2f},{r['hi']:.2f}]" for q, r in zip(qnames, rows)], fontsize=8)
    ax.set_xlabel("Base-model next-token entropy (quartile)")
    ax.set_ylabel("Tokens changed by RAD (%)")
    ax.set_ylim(0, max(1.0, max(100 * r["flip_rate"] for r in rows) * 1.25))
    ax.grid(axis="x", visible=False)

    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_path}")


def _flip_bins(rad_stats: str, n_bins: int = 4):
    """Return (qnames, rows, n, overall_flip) binning RAD steps by base entropy."""
    import pickle
    with open(rad_stats, "rb") as f:
        data = pickle.load(f)
    base_ent, flipped, gaps = [], [], []
    for d in data:
        for s in d.get("step_stats", []):
            if "base_entropy" not in s or "flipped" not in s:
                continue
            base_ent.append(s["base_entropy"]); flipped.append(s["flipped"])
            gaps.append(s.get("base_gap", float("nan")))
    if not base_ent:
        return None
    base_ent = np.asarray(base_ent); flipped = np.asarray(flipped, float); gaps = np.asarray(gaps)
    edges = np.quantile(base_ent, np.linspace(0, 1, n_bins + 1)); edges[-1] += 1e-9
    rows = []
    for b in range(n_bins):
        m = (base_ent >= edges[b]) & (base_ent < edges[b + 1])
        cnt = int(m.sum())
        rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": cnt,
                     "flip_rate": float(flipped[m].mean()) if cnt else float("nan"),
                     "mean_gap": float(np.nanmean(gaps[m])) if cnt else float("nan")})
    qnames = (["Q1", "Q2", "Q3", "Q4"] if n_bins == 4 else [f"Q{i+1}" for i in range(n_bins)])
    return qnames, rows, len(base_ent), float(flipped.mean())


def plot_calib_combined(
    greedy_stats: str, greedy_truth: str, rad_stats: str, rad_truth: str,
    flip_stats: str, n_bins: int = 4, n_ece_bins: int = 10,
    out_path: str = "results/calibration_combined.pdf", csv_out: Optional[str] = None,
) -> None:
    """Single calibration figure, two subplots (for tight paper space):
      (a) Selectivity — fraction of tokens RAD changes vs base-entropy quartile.
      (b) Reliability diagram — Greedy vs RAD, annotated with ECE and AUROC.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Error] matplotlib not installed."); return

    # --- reliability data (panel b) ---
    rel = {}
    for name, sp, tp in [("Greedy", greedy_stats, greedy_truth), ("RAD", rad_stats, rad_truth)]:
        conf_map, corr_map = _load_confidences(sp), _load_correctness(tp)
        idx = sorted(set(conf_map) & set(corr_map))
        confs = [conf_map[i] for i in idx]; corrs = [corr_map[i] for i in idx]
        ece, bins = compute_ece(confs, corrs, n_bins=n_ece_bins)
        rel[name] = {"ece": ece, "auroc": _auroc(confs, corrs), "bins": bins,
                     "acc": float(np.mean(corrs)), "mean_conf": float(np.mean(confs))}
        print(f"{name}: ECE={rel[name]['ece']:.3f} AUROC={rel[name]['auroc']:.3f} "
              f"acc={rel[name]['acc']:.3f} conf={rel[name]['mean_conf']:.3f}")

    # --- selectivity data (panel a) ---
    fb = _flip_bins(flip_stats, n_bins=n_bins)
    if fb is None:
        print(f"[Error] {flip_stats} lacks base_entropy/flipped — regenerate RAD stats "
              "with the instrumented rcd_generation."); return
    qnames, rows, n_steps, overall = fb
    print(f"Selectivity: n={n_steps} steps, overall flip={overall:.3f}; "
          + "  ".join(f"{q}={100*r['flip_rate']:.1f}%" for q, r in zip(qnames, rows)))

    if csv_out:
        _ensure_dir(csv_out)
        with open(csv_out, "w") as f:
            f.write("# panel (a) selectivity\nquartile,base_ent_lo,base_ent_hi,steps,flip_rate,mean_gap\n")
            for q, r in zip(qnames, rows):
                f.write(f"{q},{r['lo']:.4f},{r['hi']:.4f},{r['count']},{r['flip_rate']:.4f},{r['mean_gap']:.4f}\n")
            f.write("\n# panel (b) calibration summary\nmethod,ece,auroc,accuracy,mean_conf\n")
            for nm, r in rel.items():
                f.write(f"{nm},{r['ece']:.4f},{r['auroc']:.4f},{r['acc']:.4f},{r['mean_conf']:.4f}\n")
        print(f"Saved → {csv_out}")

    _use_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # (a) selectivity bars
    ax = axes[0]
    xs = np.arange(len(rows))
    bars = ax.bar(xs, [100 * r["flip_rate"] for r in rows], width=0.62,
                  color=_FLIP_BAR_COLOR, edgecolor=_FLIP_BAR_COLOR, linewidth=0.5)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=9, padding=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{q}\n[{r['lo']:.2f},{r['hi']:.2f}]" for q, r in zip(qnames, rows)], fontsize=8)
    ax.set_xlabel("Base next-token entropy (quartile, bits)")
    ax.set_ylabel("Tokens changed by RAD (%)")
    ax.set_ylim(0, max(1.0, max(100 * r["flip_rate"] for r in rows) * 1.25))
    ax.set_title("(a) Where RAD intervenes")
    ax.grid(axis="x", visible=False)

    # (b) reliability diagram
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Perfect calibration")
    style = {"Greedy": (_PAPER_COLORS[0], _PAPER_MARKERS[0]), "RAD": (_PAPER_COLORS[4], _PAPER_MARKERS[1])}
    for name, r in rel.items():
        c, mk = style[name]
        xs2 = [b["conf"] for b in r["bins"] if b["count"] > 0]
        ys2 = [b["acc"] for b in r["bins"] if b["count"] > 0]
        ax.plot(xs2, ys2, marker=mk, color=c, lw=1.8,
                label=f"{name} (ECE={r['ece']:.3f}, AUROC={r['auroc']:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (mean token prob.)")
    ax.set_ylabel("Empirical truthfulness")
    ax.set_title("(b) Reliability")
    ax.legend(loc="upper left", fontsize=8, frameon=True)

    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="DecodeHub analysis and benchmarking.")
    sub = p.add_subparsers(dest="command", required=True)

    # -- knn-latency ---------------------------------------------------------
    pk = sub.add_parser("knn-latency", help="Benchmark RAD retrieval datastore build + retrieval.")
    pk.add_argument("--base_model",       required=True)
    pk.add_argument("--embed_model_name", default="all-MiniLM-L6-v2")
    pk.add_argument("--datasets",         nargs="+", default=["truthful_qa", "wiki", "alpaca"])
    pk.add_argument("--chunk_mode",       default="chunk_8")
    pk.add_argument("--num_runs",         type=int, default=100_000)
    pk.add_argument("--result_file",      default="results/knn_latency.csv")
    pk.add_argument("--no_cosine",        action="store_true",
                    help="Use L2 (kNN-LM) instead of cosine (RAD).")

    # -- plot-ablation -------------------------------------------------------
    pa = sub.add_parser("plot-ablation", help="Training-size ablation bar chart.")
    pa.add_argument("--result_dir",  required=True)
    pa.add_argument("--method",      required=True)
    pa.add_argument("--eval_data",   required=True)
    pa.add_argument("--metric_key",  default="metric__truth_score")
    pa.add_argument("--out",         default="results/ablation.png")

    # -- embed-sensitivity ---------------------------------------------------
    pes = sub.add_parser("embed-sensitivity",
                         help="Grouped-bar chart of %%Truth/%%Info/T*I per embedder.")
    pes.add_argument("--result_file", required=True,
                     help="RAD eval JSONL, e.g. evaluation_results/truthful_qa_qwen2.5-7b_rcd.jsonl")
    pes.add_argument("--out",     default="results/embed_sensitivity.pdf")
    pes.add_argument("--csv_out", default="results/embed_sensitivity.csv")

    # -- ece -----------------------------------------------------------------
    pc = sub.add_parser("ece",
                        help="Reliability diagram + ECE for Greedy vs RAD.")
    pc.add_argument("--greedy_stats", required=True, help="Greedy _stats.pkl (--save_decoding_stats).")
    pc.add_argument("--greedy_truth", required=True, help="Greedy per-sample truth JSON (--per_sample_out).")
    pc.add_argument("--rad_stats",    required=True, help="RAD _stats.pkl.")
    pc.add_argument("--rad_truth",    required=True, help="RAD per-sample truth JSON.")
    pc.add_argument("--n_bins",       type=int, default=10)
    pc.add_argument("--out",          default="results/calibration_ece.pdf")
    pc.add_argument("--csv_out",      default="results/calibration_ece.csv")

    # -- entropy-flip --------------------------------------------------------
    pf = sub.add_parser("entropy-flip",
                        help="Token-flip rate vs base entropy (RAD selectivity).")
    pf.add_argument("--rad_stats", required=True, help="Instrumented RAD _stats.pkl.")
    pf.add_argument("--n_bins",    type=int, default=4)
    pf.add_argument("--out",       default="results/entropy_flip.pdf")
    pf.add_argument("--csv_out",   default="results/entropy_flip.csv")

    # -- calib-combined ------------------------------------------------------
    pcc = sub.add_parser("calib-combined",
                         help="Single 2-subplot calibration figure: selectivity + reliability.")
    pcc.add_argument("--greedy_stats", required=True)
    pcc.add_argument("--greedy_truth", required=True)
    pcc.add_argument("--rad_stats",    required=True)
    pcc.add_argument("--rad_truth",    required=True)
    pcc.add_argument("--flip_stats",   required=True, help="Instrumented RAD _stats.pkl (base_entropy/flipped).")
    pcc.add_argument("--n_bins",       type=int, default=4)
    pcc.add_argument("--out",          default="results/calibration_combined.pdf")
    pcc.add_argument("--csv_out",      default="results/calibration_combined.csv")

    args = p.parse_args()

    if args.command == "knn-latency":
        benchmark_knn_latency(
            base_model       = args.base_model,
            embed_model_name = args.embed_model_name,
            datasets         = args.datasets,
            chunk_mode       = args.chunk_mode,
            num_runs         = args.num_runs,
            result_file      = args.result_file,
            use_cosine       = not args.no_cosine,
        )

    elif args.command == "plot-ablation":
        plot_training_size_ablation(
            result_dir = args.result_dir,
            method     = args.method,
            eval_data  = args.eval_data,
            metric_key = args.metric_key,
            out_path   = args.out,
        )

    elif args.command == "embed-sensitivity":
        plot_embed_sensitivity(
            result_file = args.result_file,
            out_path    = args.out,
            csv_out     = args.csv_out,
        )

    elif args.command == "ece":
        plot_calibration(
            greedy_stats = args.greedy_stats,
            greedy_truth = args.greedy_truth,
            rad_stats    = args.rad_stats,
            rad_truth    = args.rad_truth,
            n_bins       = args.n_bins,
            out_path     = args.out,
            csv_out      = args.csv_out,
        )

    elif args.command == "calib-combined":
        plot_calib_combined(
            greedy_stats = args.greedy_stats, greedy_truth = args.greedy_truth,
            rad_stats    = args.rad_stats,    rad_truth    = args.rad_truth,
            flip_stats   = args.flip_stats,   n_bins       = args.n_bins,
            out_path     = args.out,          csv_out      = args.csv_out,
        )

    elif args.command == "entropy-flip":
        plot_entropy_flip(
            rad_stats = args.rad_stats,
            n_bins    = args.n_bins,
            out_path  = args.out,
            csv_out   = args.csv_out,
        )


if __name__ == "__main__":
    main()
