"""
analysis.py — Consolidated latency benchmarks and analysis utilities.

Sub-commands
------------
knn-latency     Benchmark RAD retrieval datastore retrieval time.
plot-ablation   Bar chart for training-size ablation from eval JSONL results.
entropy         Token entropy analysis across result files.

Usage examples
--------------
python analysis.py knn-latency \
    --base_model qwen2.5-7b --datasets truthful_qa wiki

python analysis.py plot-ablation \
    --result_dir results/ --method rcd --eval_data truthful_qa \
    --out results/ablation.png

python analysis.py entropy \
    --greedy_file outputs/greedy.json --method_file outputs/rcd.json
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
        nt = r.get("num_train", r.get("sample_num", 0))
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
# 4. Token entropy analysis
# ---------------------------------------------------------------------------

def analyze_token_entropy(
    result_files: list[str],
    model_name:   str,
    out_path:     str = "results/entropy.png",
) -> None:
    """Plot per-token entropy distribution for each result file."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Error] matplotlib not installed. Run: pip install matplotlib")
        return

    from transformers import AutoTokenizer
    from utils.config import get_model_path

    model_path = get_model_path(model_name)
    tokenizer  = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    for fpath in result_files:
        label   = Path(fpath).stem
        data    = json.load(open(fpath))
        entropies = []
        for item in data:
            tokens = tokenizer.encode(item.get("generated_answer", ""), add_special_tokens=False)
            if len(tokens) < 2:
                continue
            n   = len(tokens)
            p   = 1.0 / n
            ent = -p * np.log2(p) * n   # uniform approx per-token entropy
            entropies.append(ent)

        if entropies:
            ax.hist(entropies, bins=40, alpha=0.6, label=label, density=True)

    ax.set_xlabel("Per-token entropy (bits)")
    ax.set_ylabel("Density")
    ax.set_title(f"Token entropy — {model_name}")
    ax.legend()

    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")


def plot_pkl_entropy(
    pkl_files:  list[str],
    labels:     Optional[list[str]] = None,
    out_path:   str = "results/pkl_entropy.pdf",
) -> None:
    """Plot entropy distributions from _stats.pkl files produced by --save_decoding_stats.

    Each .pkl is a list of dicts:
        {question_index, step_stats: [{entropy, top_prob, token_id},...],
         mean_entropy, n_tokens}
    """
    import pickle
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Error] matplotlib not installed.")
        return

    labels = labels or [Path(f).stem for f in pkl_files]

    all_means:   list[list[float]] = []
    step_curves: list[np.ndarray]  = []

    for fpath in pkl_files:
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        means = [d["mean_entropy"] for d in data if d["n_tokens"] > 0]
        all_means.append(means)

        # Average per-position entropy across samples (truncate to shortest)
        per_step = [d["step_stats"] for d in data if d["step_stats"]]
        if per_step:
            min_len = min(len(s) for s in per_step)
            mat = np.array([[s["entropy"] for s in ss[:min_len]] for ss in per_step])
            step_curves.append(mat.mean(axis=0))
        else:
            step_curves.append(np.array([]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Histogram of mean entropy per sample
    for means, label in zip(all_means, labels):
        axes[0].hist(means, bins=30, alpha=0.6, label=label, density=True)
    axes[0].set_xlabel("Mean per-step entropy (bits)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Entropy distribution across samples")
    axes[0].legend()

    # 2. Box plot
    axes[1].boxplot(all_means, labels=labels, patch_artist=True)
    axes[1].set_ylabel("Mean per-step entropy (bits)")
    axes[1].set_title("Entropy comparison (boxplot)")

    # 3. Entropy over decoding steps
    for curve, label in zip(step_curves, labels):
        if curve.size:
            axes[2].plot(curve, label=label, alpha=0.8)
    axes[2].set_xlabel("Decoding step")
    axes[2].set_ylabel("Avg entropy (bits)")
    axes[2].set_title("Entropy vs. decoding position")
    axes[2].legend()

    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved → {out_path}")


def plot_logprob_gain_vs_entropy(
    greedy_file:  str,
    method_file:  str,
    model_name:   str,
    n_bins:       int = 5,
    out_path:     str = "results/logprob_gain.png",
) -> None:
    """Scatter / bin-average plot of log-prob gain vs. greedy entropy bins."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Error] matplotlib not installed. Run: pip install matplotlib")
        return

    from transformers import AutoTokenizer
    from utils.config import get_model_path

    model_path = get_model_path(model_name)
    tokenizer  = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    greedy_data = json.load(open(greedy_file))
    method_data = json.load(open(method_file))

    entropies, gains = [], []
    for g, m in zip(greedy_data, method_data):
        g_ans  = g.get("generated_answer", "")
        m_ans  = m.get("generated_answer", "")
        g_toks = tokenizer.encode(g_ans, add_special_tokens=False)
        m_toks = tokenizer.encode(m_ans, add_special_tokens=False)
        if not g_toks or not m_toks:
            continue
        ent  = len(g_toks)              # length as entropy proxy
        gain = len(m_toks) - len(g_toks)  # length gain as log-prob proxy
        entropies.append(ent)
        gains.append(gain)

    bins     = np.percentile(entropies, np.linspace(0, 100, n_bins + 1))
    bin_idx  = np.digitize(entropies, bins[1:-1])
    bin_means_e = [np.mean([entropies[i] for i, b in enumerate(bin_idx) if b == j]) for j in range(n_bins)]
    bin_means_g = [np.mean([gains[i]     for i, b in enumerate(bin_idx) if b == j]) for j in range(n_bins)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(entropies, gains, alpha=0.3, s=10, color="gray", label="samples")
    ax.plot(bin_means_e, bin_means_g, "o-", color="steelblue", lw=2, label="bin mean")
    ax.axhline(0, color="red", linestyle="--", lw=1)
    ax.set_xlabel("Greedy entropy (token count proxy)")
    ax.set_ylabel("Log-prob gain (method − greedy length)")
    ax.set_title(f"Log-prob gain vs. entropy — {model_name}")
    ax.legend()

    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
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

    # -- entropy -------------------------------------------------------------
    pe = sub.add_parser("entropy", help="Token entropy analysis.")
    pe.add_argument("--result_files", nargs="+", required=True)
    pe.add_argument("--model_name",   required=True)
    pe.add_argument("--out",          default="results/entropy.png")

    # -- logprob-gain --------------------------------------------------------
    pl = sub.add_parser("logprob-gain", help="Log-prob gain vs. entropy plot.")
    pl.add_argument("--greedy_file",  required=True)
    pl.add_argument("--method_file",  required=True)
    pl.add_argument("--model_name",   required=True)
    pl.add_argument("--n_bins",       type=int, default=5)
    pl.add_argument("--out",          default="results/logprob_gain.pdf")

    # -- pkl-entropy ---------------------------------------------------------
    pp = sub.add_parser("pkl-entropy",
                        help="Visualize per-step entropy from --save_decoding_stats .pkl files.")
    pp.add_argument("--pkl_files", nargs="+", required=True,
                    help="One or more _stats.pkl files (one per method).")
    pp.add_argument("--labels",    nargs="+", default=None,
                    help="Legend labels, one per pkl file.")
    pp.add_argument("--out",       default="results/pkl_entropy.pdf")

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

    elif args.command == "entropy":
        analyze_token_entropy(
            result_files = args.result_files,
            model_name   = args.model_name,
            out_path     = args.out,
        )

    elif args.command == "logprob-gain":
        plot_logprob_gain_vs_entropy(
            greedy_file = args.greedy_file,
            method_file = args.method_file,
            model_name  = args.model_name,
            n_bins      = args.n_bins,
            out_path    = args.out,
        )

    elif args.command == "pkl-entropy":
        plot_pkl_entropy(
            pkl_files = args.pkl_files,
            labels    = args.labels,
            out_path  = args.out,
        )


if __name__ == "__main__":
    main()
