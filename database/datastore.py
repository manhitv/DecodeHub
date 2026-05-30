"""
database/datastore.py — kNN-LM datastore construction and retrieval.

Workflow
--------
1. Build:   precompute_space(...)   writes .pt shard files then merges them.
2. Load:    KNNDatastore.load(...)  reads a merged .pt file and builds a FAISS index.
3. Search:  datastore.get_knn_distribution(...)  returns a token probability vector
            suitable for kNN-LM or RCD interpolation inside generation.py.

File-naming convention (must match generation.py):
    {train_data}_{model_name}_{embed_tag}_context_{chunk_mode}.pt
where embed_tag is the last component of the embed model name (no slashes).
"""

from __future__ import annotations

import os
from hashlib import md5
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.config import get_model_path, output_dir
from utils.data import extract_correct_answers, load_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed_tag(embed_model_name: str) -> str:
    """Return a filesystem-safe tag from an embed model name or path."""
    return Path(embed_model_name).name


def _hash_answer(answer: str) -> str:
    return md5(answer.encode()).hexdigest()[:8]


def _format_chat_prompt(question: str, tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return question.strip()


# ---------------------------------------------------------------------------
# Precompute (build datastore shards → merge)
# ---------------------------------------------------------------------------

def precompute_space(
    model_name: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    base_embed_model: str,
    train_data: str = "",
    store_dir: Optional[str] = None,
    num_train: int = 100,
    batch_size: int = 16,
    compute_context_types: list[str] = None,
    skip_if_exists: bool = False,
) -> None:
    """Precompute and save context embeddings + next-token logits for kNN-LM.

    Args:
        model_name:           Short alias used as part of the output filename.
        model:                Loaded HF causal-LM.
        tokenizer:            Matching tokenizer.
        base_embed_model:     SentenceTransformer model name / path.
        train_data:           Dataset key understood by load_data().
        store_dir:            Output directory (defaults to config output_dir).
        num_train:            Number of training samples to embed.
        batch_size:           Number of questions per batch for model inference.
        compute_context_types: List of context window types, e.g.
                               ['full', 'chunk_8', 'chunk_16'].
        skip_if_exists:       Skip silently if all output files already exist.
    """
    if compute_context_types is None:
        compute_context_types = ["full", "chunk_8", "chunk_16"]

    store_dir = store_dir or output_dir
    os.makedirs(store_dir, exist_ok=True)

    device    = model.device
    tag       = _embed_tag(base_embed_model)
    embed_mdl = SentenceTransformer(base_embed_model, device=device)

    # Parse context types: 'full' → ('full', None), 'chunk_8' → ('chunk_8', 8)
    context_types: list[tuple[str, Optional[int]]] = []
    for t in compute_context_types:
        if t == "full":
            context_types.append(("full", None))
        elif t.startswith("chunk_"):
            context_types.append((t, int(t.split("_")[1])))
        else:
            raise ValueError(f"Unknown context type '{t}'. Use 'full' or 'chunk_<n>'.")

    # Skip if all merged outputs already exist
    if skip_if_exists:
        all_exist = all(
            os.path.exists(_merged_path(store_dir, train_data, model_name, tag, ctype))
            for ctype, _ in context_types
        )
        if all_exist:
            print("[Skipped] All datastore files already exist.")
            return

    ds = load_data(train_data=train_data, num_train=num_train)
    results: dict[str, list] = {ctype: [] for ctype, _ in context_types}
    fallback_counter = 0

    for batch_start in tqdm(range(0, num_train, batch_size), desc="Building datastore"):
        batch_end   = min(batch_start + batch_size, num_train)
        batch_idxs  = list(range(batch_start, batch_end))

        prompts = [_format_chat_prompt(ds[i]["question"], tokenizer) for i in batch_idxs]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        for j, i in enumerate(batch_idxs):
            item       = ds[i]
            input_ids  = encoded.input_ids[j].unsqueeze(0)
            answer_list = list(dict.fromkeys(extract_correct_answers(item, train_data)))

            for answer in answer_list:
                answer_hash = _hash_answer(answer)
                record_id   = f"{i}_correct_{answer_hash}"

                target_ids  = tokenizer(answer, return_tensors="pt").input_ids.to(device)
                target_flat = target_ids.view(-1).tolist()

                try:
                    full_ids = torch.cat([input_ids, target_ids], dim=1)
                    with torch.no_grad():
                        logits = model(full_ids).logits[:, input_ids.shape[1] - 1:-1, :]
                    logits = logits[0].float().cpu()
                except Exception as exc:
                    print(f"[ERROR] {record_id}: {exc}")
                    continue

                # Build all context windows for every decoding step
                step_context_texts: list[str] = []
                step_info: list[tuple] = []
                for step in range(len(target_flat)):
                    prev_ids = torch.cat([input_ids, target_ids[:, :step]], dim=1)[0]
                    for ctype, chunk in context_types:
                        ctx_ids = prev_ids if chunk is None else prev_ids[-chunk:]
                        text    = tokenizer.decode(ctx_ids, skip_special_tokens=True)
                        step_context_texts.append(text)
                        step_info.append((step, ctype, text))

                # Batch-encode all contexts for this sample
                try:
                    embeds = embed_mdl.encode(
                        step_context_texts, convert_to_tensor=True, batch_size=64
                    )
                    embeds = F.normalize(embeds, dim=-1).float().cpu()
                except Exception:
                    dim    = embed_mdl.get_sentence_embedding_dimension()
                    embeds = torch.zeros(len(step_context_texts), dim).float()
                    fallback_counter += len(step_context_texts)

                for idx, (step, ctype, text) in enumerate(step_info):
                    results[ctype].append({
                        "record_id":         record_id,
                        "question_id":       i,
                        "question":          item["question"],
                        "answer":            answer,
                        "step":              step,
                        "context_embedding": embeds[idx],
                        "logits":            logits[step],
                        "context_text":      text,
                        "next_token_id":     target_flat[step],
                    })

        # Flush shards to disk after each batch
        for ctype, _ in context_types:
            shard_path = _shard_path(
                store_dir, train_data, model_name, tag, batch_start, batch_end, ctype
            )
            torch.save(results[ctype], shard_path)
            results[ctype].clear()

    # Merge shards into final files
    for ctype, _ in context_types:
        merged = []
        for fname in sorted(os.listdir(store_dir)):
            prefix = f"{train_data}_{model_name}_{tag}_batch_"
            suffix = f"_context_{ctype}.pt"
            if fname.startswith(prefix) and fname.endswith(suffix):
                merged.extend(torch.load(os.path.join(store_dir, fname), weights_only=True))
                os.remove(os.path.join(store_dir, fname))

        final = _merged_path(store_dir, train_data, model_name, tag, ctype)
        torch.save(merged, final)
        print(f"[SAVED] {final} — {len(merged)} entries")

    if fallback_counter:
        print(f"[Info] Zero-embedding fallbacks: {fallback_counter}")


def _shard_path(store_dir, train_data, model_name, tag, start, end, ctype):
    return os.path.join(
        store_dir,
        f"{train_data}_{model_name}_{tag}_batch_{start}_{end}_context_{ctype}.pt",
    )


def _merged_path(store_dir, train_data, model_name, tag, ctype):
    return os.path.join(store_dir, f"{train_data}_{model_name}_{tag}_context_{ctype}.pt")


# ---------------------------------------------------------------------------
# KNNDatastore — load + FAISS index + retrieval
# ---------------------------------------------------------------------------

class KNNDatastore:
    """Wraps a pre-built .pt datastore file and a FAISS index for kNN-LM / RCD.

    Usage
    -----
    ds = KNNDatastore.load(train_data, model_name, embed_model, store_dir)
    p_knn = ds.get_knn_distribution(query_text, vocab_size, k=64, temperature=1.0)
    """

    def __init__(self, data: list[dict], embed_model: SentenceTransformer, use_cosine: bool = False):
        self._data        = data
        self._embed_mdl   = embed_model
        self._use_cosine  = use_cosine
        self._index       = self._build_index(data, use_cosine=use_cosine)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_index(data: list[dict], use_cosine: bool = False) -> faiss.Index:
        embeddings = torch.stack([d["context_embedding"] for d in data])
        if use_cosine:
            embeddings = F.normalize(embeddings, dim=-1)
        emb_np = embeddings.cpu().numpy().astype("float32")
        if use_cosine:
            index = faiss.IndexFlatIP(emb_np.shape[1])
        else:
            index = faiss.IndexFlatL2(emb_np.shape[1])
        index.add(emb_np)
        return index

    @classmethod
    def load(
        cls,
        train_data:       str,
        model_name:       str,
        embed_model_name: str,
        store_dir:        Optional[str] = None,
        device:           str           = "cpu",
        use_gpu:          bool          = False,
        use_cosine:       bool          = False,
    ) -> "KNNDatastore":
        """Find the best matching .pt file and return a ready KNNDatastore.

        Tries chunk modes in order: full → chunk_16 → chunk_8.
        """
        store_dir = store_dir or output_dir
        tag       = _embed_tag(embed_model_name)

        data_path = None
        for chunk_mode in ("full", "chunk_16", "chunk_8"):
            candidate = _merged_path(store_dir, train_data, model_name, tag, chunk_mode)
            if os.path.exists(candidate):
                data_path = candidate
                break

        if data_path is None:
            raise FileNotFoundError(
                f"No kNN-LM datastore found for train_data='{train_data}', "
                f"model='{model_name}', embed='{embed_model_name}' in '{store_dir}'.\n"
                f"Run precompute_space(...) first."
            )

        data       = torch.load(data_path, weights_only=True)
        embed_mdl  = SentenceTransformer(embed_model_name, device=device)
        instance   = cls.__new__(cls)
        instance._data      = data
        instance._embed_mdl = embed_mdl
        instance._use_cosine = use_cosine
        instance._index     = cls._build_index(data, use_cosine=use_cosine)

        if use_gpu and faiss.get_num_gpus() > 0:
            res             = faiss.StandardGpuResources()
            instance._index = faiss.index_cpu_to_gpu(res, 0, instance._index)

        return instance

    # ------------------------------------------------------------------
    # Retrieval — kNN-LM
    # ------------------------------------------------------------------

    def _embed_query(self, text: str) -> np.ndarray:
        emb = self._embed_mdl.encode(text, convert_to_tensor=True).float()
        if self._use_cosine:
            emb = F.normalize(emb, dim=-1)
        return emb.cpu().numpy().astype("float32").reshape(1, -1)

    def search(self, query_text: str, k: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances, indices) arrays of shape (k,)."""
        q   = self._embed_query(query_text)
        k   = min(k, len(self._data))
        D, I = self._index.search(q, k)
        return D[0], I[0]

    def get_knn_distribution(
        self,
        query_text: str,
        vocab_size:  int,
        k:           int   = 64,
        temperature: float = 1.0,
        device:      str   = "cpu",
    ) -> torch.Tensor:
        """Return a kNN token probability vector of shape (vocab_size,).

        Uses distance-weighted softmax over the retrieved next-token ids.
        """
        dists, idxs = self.search(query_text, k)
        p = torch.zeros(vocab_size, device=device)
        for dist, idx in zip(dists, idxs):
            token_id = self._data[idx]["next_token_id"]
            weight   = torch.exp(torch.tensor(-dist / temperature, device=device))
            p[token_id] += weight
        total = p.sum()
        return p / (total + 1e-8)

    # ------------------------------------------------------------------
    # Retrieval — RCD (Retrieval-Augmented Contrastive Decoding)
    # ------------------------------------------------------------------

    def retrieve_passages(self, query_text: str, k: int = 5) -> list[dict]:
        """Return the top-k datastore entries closest to *query_text*.

        Each entry dict contains at least: question, answer, context_text.
        Used by RCD to build a retrieved-passage contrastive baseline.
        """
        _, idxs = self.search(query_text, k)
        return [self._data[i] for i in idxs]


# ---------------------------------------------------------------------------
# CLI entry point — build a kNN-LM datastore from the command line
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Build a kNN-LM datastore.")
    parser.add_argument("--base_model",       type=str, required=True,
                        help="Model alias from config.yaml or a full HF repo id.")
    parser.add_argument("--base_embed_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--train_data",       type=str, default="truthful_qa")
    parser.add_argument("--num_train",        type=int, default=100)
    parser.add_argument("--batch_size",       type=int, default=16)
    parser.add_argument("--chunk_size",       type=int, default=0,
                        help="Token chunk size (0 = full context).")
    parser.add_argument("--store_dir",        type=str, default=None,
                        help="Output directory (defaults to config output_dir).")
    parser.add_argument("--skip_if_exists",   action="store_true")
    args = parser.parse_args()

    model_path = get_model_path(args.base_model)
    use_fast   = "falcon3" not in model_path.lower()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side="left", trust_remote_code=True, use_fast=use_fast
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    context_types = [f"chunk_{args.chunk_size}"] if args.chunk_size > 0 else ["full"]

    t0 = time.time()
    precompute_space(
        model_name           = args.base_model,
        model                = model,
        tokenizer            = tokenizer,
        base_embed_model     = args.base_embed_model,
        train_data           = args.train_data,
        store_dir            = args.store_dir,
        num_train            = args.num_train,
        batch_size           = args.batch_size,
        compute_context_types = context_types,
        skip_if_exists       = args.skip_if_exists,
    )
    print(f"Done in {time.time() - t0:.1f}s")
