<div align="center">
<img src="assets/decodehub_logo.svg" height=150 alt="DecodeHub — RAD">
  <h1><b> RAD: Retrieval-Augmented Decoding for Truthful Generation </b></h1>
  <p><i>A unified, modular framework for hallucination-reduction decoding in large language models.</i></p>
</div>

<div align="center">

[![Venue](https://img.shields.io/badge/ECML%20PKDD-2026-blue?logo=springer&logoColor=white)](https://link.springer.com/chapter/10.1007/978-3-032-37673-2_2)
[![arXiv](https://img.shields.io/badge/arXiv-2508.02184-b31b1b)](https://arxiv.org/pdf/2508.02184)
[![PyTorch](https://img.shields.io/badge/Powered_by-PyTorch-ee4c2c)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/Retrieval-FAISS-blue)](https://github.com/facebookresearch/faiss)

🚀 [**Install**](#install) **|** 🔧 [**Usage**](#usage) **|** 🧪 [**Reproduce**](#reproduce) **|** 🎯 [**Benchmarks**](#bench) **|** 🧠 [**Baselines**](#baselines) **|** 📂 [**Structure**](#structure)

</div>

**RAD — Retrieval-Augmented Decoding** improves the truthfulness of LLM open-ended generation in a **single forward pass, without retraining, no auxiliary model, and no model-specific states**. From as few as **10 annotated examples** it builds a compact
*grounding space* of `(context embedding, next-token logits)` pairs; at each decoding step it retrieves contexts above a cosine-similarity threshold and fuses their similarity-weighted logits into the model's own.

<div align="center">
<img src="assets/rad_overview.png" alt="RAD overview" width="100%">
</div>

📜 **Paper:** [*Retrieval-Augmented Decoding for Improving Truthfulness in Open-Ended Generation*](https://link.springer.com/chapter/10.1007/978-3-032-37673-2_2) (ECML PKDD 2026)

> ```bibtex
> @inproceedings{nguyen2026rad,
>  title     = {Retrieval-Augmented Decoding for Improving Truthfulness in Open-Ended Generation},
>  author    = {Nguyen, Manh and Gupta, Sunil and Le, Hung},
>  booktitle = {Machine Learning and Knowledge Discovery in Databases. Research Track},
>  year      = {2027},
>  publisher = {Springer Nature Switzerland},
>  pages     = {19--36}
> }
> ```

---

## <a name="install"></a> 🚀 Installation

```bash
git clone <your-repo-url> RAD && cd RAD
conda create -n decodehub python=3.10 -y && conda activate decodehub
pip install -r requirements.txt          # no GPU? swap faiss-gpu → faiss-cpu
```

RAD inference is fully local; only **evaluation** uses the Cohere (or Gemini) API. Set keys via
`cp local/keys.env.template local/keys.env` or `export COHERE_API_KEY=...` (`GEMINI_API_KEY`
optional). Use `--run_only` to skip evaluation — no key needed.

`config.yaml` holds model aliases and I/O paths (override with `DECODEHUB_OUTPUT`,
`DECODEHUB_DATA`, …). Benchmark datasets are fetched from HuggingFace on first use;
`output/`, `results/`, `evaluation_results/` are created automatically.

---

## <a name="usage"></a> 🔧 Usage

`run.py` is the single entry point; `--decoding_method` picks the method (**RAD = `rcd`**).

```bash
# 1) build the grounding space (context embeddings + next-token logits)
python -m database.datastore --base_model qwen2.5-7b --train_data wiki \
  --num_train 100 --chunk_size 8

# 2) run RAD inference + evaluation
python run.py --base_model qwen2.5-7b --decoding_method rcd \
  --train_data wiki --eval_data wiki --num_train 100 \
  --embed_model_name all-MiniLM-L6-v2 --eval_metric factuality \
  --configs_json '[{"shaping_mode":"linear","alpha":0.5,"sim_threshold":0.7,"agg_mode":"weighted"}]'
```

With no `--configs_json`, RAD uses the built-in default for the `(rcd, <dataset>)` pair.
`bash scripts/validate.sh` runs a quick smoke test (no API key needed).

| Flag | Description |
|------|-------------|
| `--decoding_method` | `rcd` (**RAD**), `greedy`, `cad`, `dola`, `instructive`, `kNN-ICL`, `knn_lm` |
| `--base_model`      | Alias from `config.yaml` (e.g. `qwen2.5-7b`), HF repo id, or local path |
| `--eval_data`       | `truthful_qa`, `wiki`, `alpaca`, `halu_dia`, `halu_sum` |
| `--train_data`      | Grounding corpus for RAD / kNN-LM (defaults to `--eval_data`) |
| `--num_train`       | Number of grounding instances (default 100) |
| `--configs_json`    | Per-run RAD overrides: `alpha`, `sim_threshold`, `chunk_size`, `agg_mode`, `shaping_mode`, `exact_match` |
| `--eval_metric`     | `factuality` (%Truth, %Info, T\*I) or `halu_rate` |
| `--evaluation_type` | `cohere` (`command-a-03-2025`) or `gemini` (`gemini-2.0-flash`) |
| `--run_only`        | Skip API evaluation; save responses + latency stats only |
| `--embed_model_name`| Sentence embedder for the grounding space (default `all-MiniLM-L6-v2`) |

Outputs: generations → `output/`, eval results → `results/` and `evaluation_results/`,
timing → `<output>_timing.json`.

**RAD defaults** (`run.py::_build_rcd_cfg`): embedder `all-MiniLM-L6-v2`, chunk `M=8`,
threshold `tau=0.7` (`0.8` for Alpaca), weight `alpha=0.5`, similarity-weighted aggregation,
`N=100` grounding instances, FAISS cosine index. Paper runs on a single H100 80GB.

---

## <a name="reproduce"></a> 🧪 Reproducing Paper Results

Each table/figure maps to one script. All require `COHERE_API_KEY`.

| Paper | Script |
|-------|--------|
| Tables 1–2 (main QA + HaluEval) | `scripts/run_main_table.sh` |
| Table 3 (out-of-distribution)   | `scripts/run_ood.sh` |
| Table 4 (grounding sizes/latency) + Fig 6 | `scripts/run_analysis.sh` |
| Table 5 (embedding model)       | `scripts/run_embed_sensitivity.sh` |
| Table 6 + Figs 3–4 (ablations)  | `scripts/run_ablations.sh` |
| Fig 5 (calibration)             | `scripts/run_calibration.sh` |

Datasets download from HuggingFace on first use; the grounding space is built once per
`(model, dataset)` and cached. Large `.pt` grounding files go to `$DECODEHUB_OUTPUT`
(default `/weka/$USER/decodehub/output`) to keep them off the home partition.
The embedding-sensitivity and calibration scripts cache the extra sentence embedders via
`bash scripts/prefetch_embedders.sh` (run once on a node with internet).

---

## <a name="bench"></a> 🎯 Benchmarks

**Models:** `qwen2.5-3b`, `qwen2.5-7b`, `mistral2-7b`, `gemma2-9b`.

| `--eval_data` | Benchmark | Metric |
|---------------|-----------|--------|
| `truthful_qa` | TruthfulQA — common misconceptions | `factuality` (%Truth, %Info, T\*I) |
| `wiki`        | WikiQA — Wikipedia-grounded QA     | `factuality` |
| `alpaca`      | Alpaca — reasoning / instructions  | `factuality` |
| `halu_dia`    | HaluEval — dialogue                | `halu_rate` (lower is better) |
| `halu_sum`    | HaluEval — summarization           | `halu_rate` |

---

## <a name="baselines"></a> 🧠 Baselines

Six single-pass baselines, all via `run.py`. Paper→code names: **ID = `instructive`**,
**KATE = `kNN-ICL`**, **RAD = `rcd`**.

| Method | `--decoding_method` | Idea |
|--------|---------------------|------|
| Greedy | `greedy` | Standard argmax decoding |
| CAD    | `cad` | Contrasts full-context vs. bare-question logits |
| DoLa   | `dola` | Contrasts later vs. earlier transformer layers |
| ID     | `instructive` | Contrasts standard vs. adversarial-prompt logits |
| KATE   | `kNN-ICL` | Retrieves top-k similar Q&A pairs into the prompt |
| kNN-LM | `knn_lm` | Interpolates the LM distribution with a kNN datastore |
| **RAD (Ours)** | `rcd` | Retrieval-augmented logit shaping over a grounding space |

---

## <a name="structure"></a> 📂 Project Structure

```text
RAD/
├── run.py                 # Inference + evaluation entry point
├── analysis.py            # Latency benchmark + ablation / calibration plots
├── eval_from_file.py      # Re-evaluate a pre-generated output JSON
├── eval_open_longform.py  # ROUGE-L / BERTScore for long-form outputs
├── config.yaml            # Model aliases and I/O paths
├── scripts/               # Per-table/figure reproduction + validation scripts
├── database/datastore.py  # Grounding space: precompute + FAISS index + retrieval
└── utils/                 # decoding (generation.py), data, prompts, evaluation
```

Released under the [MIT License](LICENSE). Built with
[PyTorch](https://pytorch.org/), [Transformers](https://github.com/huggingface/transformers),
[FAISS](https://github.com/facebookresearch/faiss), and [Cohere](https://cohere.com/) for evaluation.
