"""
generation.py — Decoding-time intervention techniques
Supports: RAD (rcd), kNN-LM, Contrastive Decoding (CAD),
          Instructive Decoding (ID), DoLa, Greedy, ICL, KAPING, kNN-ICL
"""

from __future__ import annotations

import os
from typing import Optional

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from database.datastore import KNNDatastore
from utils.config import output_dir
from utils.prompt import PromptTemplateLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_from_path(
    base_model_path: str,
    device: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a HuggingFace causal-LM and its tokenizer."""
    use_fast = "falcon3" not in base_model_path.lower()
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        padding_side="left",
        trust_remote_code=True,
        use_fast=use_fast,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if "gemma3" in base_model_path.lower():
        from transformers import Gemma3ForCausalLM
        model = Gemma3ForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.float16,
            device_map=device, trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.float16,
            device_map=device, trust_remote_code=True,
        )
    return model, tokenizer


def _attention_mask_for(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    return (input_ids != pad_token_id).long().to(input_ids.device)


# ---------------------------------------------------------------------------
# kNN-LM  (backed by KNNDatastore)
# ---------------------------------------------------------------------------

_DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


def _compute_step_stats(logits: torch.Tensor, chosen_token: int) -> dict:
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = float(-(probs * torch.log2(probs + 1e-10)).sum())
    return {"entropy": entropy, "top_prob": float(probs.max()), "token_id": chosen_token}


def _compute_rcd_step_stats(
    base_logit: torch.Tensor, adjusted_logit: torch.Tensor, chosen_token: int
) -> dict:
    """Per-step stats for RAD, recording the BASE (pre-fusion) distribution and
    whether the retrieved term flipped the arg max. Used by the selectivity /
    'where RAD intervenes' analysis. Keeps the base keys (entropy/top_prob/
    token_id) of _compute_step_stats so the entropy/ECE tools still work."""
    base_p = torch.softmax(base_logit.float(), dim=-1)
    adj_p  = torch.softmax(adjusted_logit.float(), dim=-1)
    base_entropy = float(-(base_p * torch.log2(base_p + 1e-10)).sum())
    adj_entropy  = float(-(adj_p * torch.log2(adj_p + 1e-10)).sum())
    top2 = torch.topk(base_p, 2).values
    return {
        "entropy":       adj_entropy,            # final (post-fusion) — back-compat
        "top_prob":      float(adj_p.max()),
        "token_id":      chosen_token,
        "base_entropy":  base_entropy,
        "base_top_prob": float(base_p.max()),
        "base_gap":      float(top2[0] - top2[1]),   # top1 - top2 of base
        "flipped":       int(int(base_logit.argmax()) != int(adjusted_logit.argmax())),
    }


def _load_knn_datastore(
    train_data:       str,
    model_name:       str,
    embed_model_name: str = _DEFAULT_EMBED_MODEL,
    use_gpu:          bool = True,
) -> KNNDatastore:
    """Load a KNNDatastore; use GPU FAISS index when available."""
    return KNNDatastore.load(
        train_data       = train_data,
        model_name       = model_name,
        embed_model_name = embed_model_name,
        store_dir        = output_dir,
        use_gpu          = use_gpu,
    )


# ---------------------------------------------------------------------------
# kNN-LM decoding  (probability interpolation, backed by KNNDatastore)
# ---------------------------------------------------------------------------

def knn_lm_decoding(
    model: AutoModelForCausalLM,
    model_name: str,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,                 # (1, seq_len)
    train_data: str,
    num_train: int,
    max_new_tokens: int = 128,
    alpha: float = 0.5,
    attention_mask: Optional[torch.Tensor] = None,
    # Pre-built kNN-LM datastore (shared across samples to avoid re-loading)
    knn_datastore: Optional["KNNDatastore"] = None,
    collect_stats: bool = False,
) -> "str | tuple[str, list]":
    """Token-by-token kNN-LM: interpolate the LM distribution with a
    distance-weighted kNN distribution retrieved from the KNNDatastore."""
    device = next(model.parameters()).device
    generated = input_ids.to(device).clone()
    if attention_mask is None:
        attention_mask = _attention_mask_for(generated, tokenizer.pad_token_id)

    # Lazy-load the datastore if not provided externally
    if knn_datastore is None:
        knn_datastore = _load_knn_datastore(
            train_data, model_name, use_gpu=torch.cuda.is_available()
        )

    step_stats: list[dict] = []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=generated,
                attention_mask=attention_mask,
                use_cache=False,
            )
        base_logit = outputs.logits[0, -1, :]

        vocab_size = getattr(
            model.config, "vocab_size",
            model.get_input_embeddings().num_embeddings,
        )
        context_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        k = 2048
        p_knn = knn_datastore.get_knn_distribution(
            context_text, vocab_size, k=k, temperature=1.0, device=str(device)
        )
        p_lm = torch.softmax(base_logit, dim=-1)
        adjusted_logits = torch.log((1 - alpha) * p_lm + alpha * p_knn + 1e-8)

        best_token = torch.argmax(adjusted_logits)
        if collect_stats:
            step_stats.append(_compute_step_stats(adjusted_logits, int(best_token.item())))
        generated = torch.cat([generated, best_token.view(1, 1)], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1
        )

        if tokenizer.eos_token_id is not None and best_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
    return (text, step_stats) if collect_stats else text


def knn_lm_decoding_batch(
    model: AutoModelForCausalLM,
    model_name: str,
    tokenizer: AutoTokenizer,
    input_ids_batch: torch.Tensor,
    train_data: str,
    num_train: int,
    max_new_tokens: int = 128,
    alpha: float = 0.5,
    collect_stats: bool = False,
) -> "list[str] | tuple[list[str], list]":
    device = next(model.parameters()).device
    input_ids_batch = input_ids_batch.to(device)

    # Load the kNN-LM datastore once and share it across all samples in the batch
    knn_datastore = _load_knn_datastore(
        train_data, model_name, use_gpu=torch.cuda.is_available()
    )

    all_outputs, all_stats = [], []
    for i in range(input_ids_batch.size(0)):
        input_ids = input_ids_batch[i].unsqueeze(0)
        attention_mask = _attention_mask_for(input_ids, tokenizer.pad_token_id)
        result = knn_lm_decoding(
            model=model,
            model_name=model_name,
            tokenizer=tokenizer,
            input_ids=input_ids,
            train_data=train_data,
            num_train=num_train,
            max_new_tokens=max_new_tokens,
            alpha=alpha,
            attention_mask=attention_mask,
            knn_datastore=knn_datastore,
            collect_stats=collect_stats,
        )
        if collect_stats:
            output_text, sample_stats = result
            all_stats.append(sample_stats)
        else:
            output_text = result
        all_outputs.append(output_text)
    return (all_outputs, all_stats) if collect_stats else all_outputs


# ---------------------------------------------------------------------------
# RCD — Retrieval-augmented Contrastive Decoding
# ---------------------------------------------------------------------------

def _get_rcd_logits(
    sims: list[float],
    idxs: list[int],
    data: list[dict],
    vocab_size: int,
    top_n: Optional[int],
    sim_threshold: float,
    mode: str,
    device,
) -> torch.Tensor:
    """Weighted or mean aggregation of retrieved full-vocab logit vectors."""
    filtered_sims, filtered_idxs = [], []
    for s, i in zip(sims, idxs):
        if s > sim_threshold:
            filtered_sims.append(s)
            filtered_idxs.append(i)

    if not filtered_idxs:
        return torch.zeros(vocab_size, dtype=torch.float32, device=device)

    n = top_n if top_n is not None else len(filtered_idxs)
    use_idxs   = filtered_idxs[:n]
    use_sims   = filtered_sims[:n]
    logit_vecs = [data[i]["logits"].to(dtype=torch.float32, device=device) for i in use_idxs]

    if mode == "mean":
        return torch.stack(logit_vecs).mean(dim=0)

    weights = torch.tensor(use_sims, dtype=torch.float32, device=device)
    weights = weights / weights.sum()
    return (torch.stack(logit_vecs) * weights.unsqueeze(1)).sum(dim=0)


def rcd_generation(
    model: AutoModelForCausalLM,
    model_name: str,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    knn_datastore: "KNNDatastore",
    max_new_tokens: int = 128,
    alpha: float = 0.5,
    shaping_mode: str = "linear",
    top_n: Optional[int] = None,
    sim_threshold: float = 0.7,
    agg_mode: str = "weighted",
    chunk_size: Optional[int] = 8,
    exact_match: str = "",
    attention_mask: Optional[torch.Tensor] = None,
    collect_stats: bool = False,
) -> "str | tuple[str, list]":
    """Token-by-token RCD: combines base logits with retrieved full-vocab logits."""
    device    = next(model.parameters()).device
    generated = input_ids.to(device).clone()
    if attention_mask is None:
        attention_mask = _attention_mask_for(generated, tokenizer.pad_token_id)

    data       = knn_datastore._data
    vocab_size = getattr(model.config, "vocab_size", model.get_input_embeddings().num_embeddings)

    step_stats: list[dict] = []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=generated, attention_mask=attention_mask, use_cache=False)
        base_logit = outputs.logits[0, -1, :]

        if shaping_mode == "greedy":
            adjusted_logits = base_logit
        else:
            ctx_ids  = generated[0] if chunk_size is None else generated[0][-chunk_size:]
            ctx_text = tokenizer.decode(ctx_ids, skip_special_tokens=True)

            sims, idxs = knn_datastore.search(ctx_text, k=min(top_n or 2048, 2048))
            sims, idxs = sims.tolist(), idxs.tolist()

            if exact_match:
                filtered = [
                    (s, i) for s, i in zip(sims, idxs)
                    if (exact_match == "ignore") != (data[i].get("context_text", "").strip() == ctx_text.strip())
                ]
                sims, idxs = ([x[0] for x in filtered], [x[1] for x in filtered]) if filtered else ([], [])

            pos_logit = _get_rcd_logits(sims, idxs, data, vocab_size, top_n, sim_threshold, agg_mode, device)

            if shaping_mode == "linear":
                adjusted_logits = base_logit + alpha * pos_logit
            elif shaping_mode == "delta":
                adjusted_logits = (1 - alpha) * base_logit + alpha * pos_logit
            elif shaping_mode == "prob_interp":
                p_lm  = torch.softmax(base_logit, dim=-1)
                p_ret = torch.softmax(pos_logit,  dim=-1)
                adjusted_logits = torch.log((1 - alpha) * p_lm + alpha * p_ret + 1e-8)
            else:
                raise ValueError(f"Unknown RCD shaping_mode: {shaping_mode!r}")

        best_token = torch.argmax(adjusted_logits)
        if collect_stats:
            step_stats.append(
                _compute_rcd_step_stats(base_logit, adjusted_logits, int(best_token.item()))
            )
        generated  = torch.cat([generated, best_token.view(1, 1)], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1
        )
        if tokenizer.eos_token_id is not None and best_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
    return (text, step_stats) if collect_stats else text


def rcd_generation_batch(
    model: AutoModelForCausalLM,
    model_name: str,
    tokenizer: AutoTokenizer,
    input_ids_batch: torch.Tensor,
    knn_datastore: "KNNDatastore",
    max_new_tokens: int = 128,
    alpha: float = 0.5,
    shaping_mode: str = "linear",
    top_n: Optional[int] = None,
    sim_threshold: float = 0.7,
    agg_mode: str = "weighted",
    chunk_size: Optional[int] = 8,
    exact_match: str = "",
    collect_stats: bool = False,
) -> "list[str] | tuple[list[str], list]":
    device = next(model.parameters()).device
    input_ids_batch = input_ids_batch.to(device)
    results, all_stats = [], []
    for i in range(input_ids_batch.size(0)):
        ids  = input_ids_batch[i].unsqueeze(0)
        mask = _attention_mask_for(ids, tokenizer.pad_token_id)
        result = rcd_generation(
            model=model,
            model_name=model_name,
            tokenizer=tokenizer,
            input_ids=ids,
            knn_datastore=knn_datastore,
            max_new_tokens=max_new_tokens,
            alpha=alpha,
            shaping_mode=shaping_mode,
            top_n=top_n,
            sim_threshold=sim_threshold,
            agg_mode=agg_mode,
            chunk_size=chunk_size,
            exact_match=exact_match,
            attention_mask=mask,
            collect_stats=collect_stats,
        )
        if collect_stats:
            text, sample_stats = result
            all_stats.append(sample_stats)
        else:
            text = result
        results.append(text)
    return (results, all_stats) if collect_stats else results


# ---------------------------------------------------------------------------
# Instructive Decoding (ID)
# ---------------------------------------------------------------------------

def instructive_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    noisy_input_ids: torch.Tensor,
    eta: float = 0.3,
    max_new_tokens: int = 32,
    model_name: Optional[str] = None,
    train_data: Optional[str] = None,
    alpha: float = 0.5,
    num_train: int = 100,
) -> str:
    device = next(model.parameters()).device
    generated = input_ids.clone().to(device)
    noisy_generated = noisy_input_ids.clone().to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            next_logits = model(generated).logits[0, -1, :]
            noisy_logits = model(noisy_generated).logits[0, -1, :]

        adjusted_logits = next_logits - eta * noisy_logits

        best_token = torch.argmax(adjusted_logits)
        generated = torch.cat([generated, best_token.view(1, 1)], dim=1)
        noisy_generated = torch.cat([noisy_generated, best_token.view(1, 1)], dim=1)

        if tokenizer.eos_token_id is not None and best_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)


def instructive_generation_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids_batch: torch.Tensor,
    noisy_input_ids_batch: torch.Tensor,
    eta: float = 0.3,
    max_new_tokens: int = 32,
    model_name: Optional[str] = None,
    train_data: Optional[str] = None,
    num_train: int = 100,
    alpha: float = 0.5,
) -> list[str]:
    responses = []
    for input_ids, noisy_input_ids in zip(input_ids_batch, noisy_input_ids_batch):
        responses.append(instructive_generation(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids.unsqueeze(0),
            noisy_input_ids=noisy_input_ids.unsqueeze(0),
            eta=eta,
            max_new_tokens=max_new_tokens,
            model_name=model_name,
            train_data=train_data,
            alpha=alpha,
            num_train=num_train,
        ))
    return responses


# ---------------------------------------------------------------------------
# Contrastive Decoding (CAD)
# ---------------------------------------------------------------------------

def cad_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    noisy_input_ids: torch.Tensor,
    eta: float = 0.5,
    max_new_tokens: int = 32,
) -> str:
    device = next(model.parameters()).device
    generated = input_ids.clone().to(device)
    noisy_generated = noisy_input_ids.clone().to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            next_logits = model(generated).logits[:, -1, :]
            noisy_logits = model(noisy_generated).logits[:, -1, :]

        adjusted_logits = next_logits + eta * (next_logits - noisy_logits)
        best_token = torch.argmax(F.softmax(adjusted_logits, dim=-1), dim=-1)

        generated = torch.cat([generated, best_token.unsqueeze(0)], dim=1)
        noisy_generated = torch.cat([noisy_generated, best_token.unsqueeze(0)], dim=1)

        if tokenizer.eos_token_id is not None and best_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)


def cad_generation_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids_batch: torch.Tensor,
    noisy_input_ids_batch: torch.Tensor,
    eta: float = 0.5,
    max_new_tokens: int = 32,
) -> list[str]:
    responses = []
    for input_ids, noisy_input_ids in zip(input_ids_batch, noisy_input_ids_batch):
        responses.append(cad_generation(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids.unsqueeze(0),
            noisy_input_ids=noisy_input_ids.unsqueeze(0),
            eta=eta,
            max_new_tokens=max_new_tokens,
        ))
    return responses


# ---------------------------------------------------------------------------
# Base Generator
# ---------------------------------------------------------------------------

_DECODING_METHODS_NEEDING_NOISY = {"instructive", "cad"}
_DECODING_METHODS_USING_RCD     = {"rcd"}
_DECODING_METHODS_HF_GENERATE   = {"greedy", "dola", "icl", "KAPING", "kNN-ICL"}


class BaseGenerator:
    """
    Thin wrapper around a single HuggingFace model that dispatches
    to the correct decoding function.
    """

    def __init__(
        self,
        model_path: str,
        decoding_method: str,
        device: str = "cuda:0",
        generation_config: Optional[dict] = None,
        decoding_config: Optional[dict] = None,
    ):
        self.base_model_path = model_path
        self.base_model, self.base_tokenizer = load_model_from_path(model_path, device)
        self.base_model.eval()
        self.device = 0 if device == "auto" else device

        generation_config = generation_config or {}
        generation_config.setdefault("max_new_tokens", 256)
        self.generation_config = GenerationConfig(**generation_config)
        self.max_new_tokens = generation_config["max_new_tokens"]

        self.decoding_config = decoding_config or {}
        self.decoding_method = decoding_method

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inference_on_data(
        self,
        prompts: list[str],
        batch_size: int = 16,
        noisy_prompts: Optional[list[str]] = None,
        stats_path: Optional[str] = None,
    ) -> list[str]:
        collect = stats_path is not None
        responses, all_stats = [], []
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch       = prompts[i : i + batch_size]
            noisy_batch = noisy_prompts[i : i + batch_size] if noisy_prompts else None
            out = self._run_batch(batch, noisy_batch, collect_stats=collect)
            if collect:
                batch_responses, batch_stats = out
                all_stats.extend(batch_stats)
            else:
                batch_responses = out
            responses.extend(batch_responses)

        if collect and stats_path:
            import pickle
            annotated = [
                {
                    "question_index": idx,
                    "step_stats": ss,
                    "mean_entropy": float(sum(s["entropy"] for s in ss) / len(ss)) if ss else 0.0,
                    "n_tokens": len(ss),
                }
                for idx, ss in enumerate(all_stats)
            ]
            os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
            with open(stats_path, "wb") as f:
                pickle.dump(annotated, f)
            print(f"[Stats saved] {stats_path}")

        return responses

    def inference_one_sample(
        self,
        prompt: str,
        noisy_prompt: Optional[str] = None,
    ) -> str:
        return self.inference_on_data(
            [prompt],
            batch_size=1,
            noisy_prompts=[noisy_prompt] if noisy_prompt else None,
        )[0]

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        enc = self.base_tokenizer(texts, return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in enc.items()}

    def _run_batch(
        self,
        prompts: list[str],
        noisy_prompts: Optional[list[str]],
        collect_stats: bool = False,
    ) -> "list[str] | tuple[list[str], list]":
        base_inputs = self._tokenize(prompts)

        if self.decoding_method in _DECODING_METHODS_NEEDING_NOISY:
            if noisy_prompts is None:
                raise ValueError(
                    f"Decoding method '{self.decoding_method}' requires noisy_prompts."
                )
            noisy_inputs = self._tokenize(noisy_prompts)

            if self.decoding_method == "instructive":
                responses = instructive_generation_batch(
                    model=self.base_model,
                    tokenizer=self.base_tokenizer,
                    input_ids_batch=base_inputs["input_ids"],
                    noisy_input_ids_batch=noisy_inputs["input_ids"],
                    eta=self.decoding_config.get("eta", 0.3),
                    max_new_tokens=self.max_new_tokens,
                    model_name=self.decoding_config.get("model_name"),
                    train_data=self.decoding_config.get("train_data"),
                    num_train=self.decoding_config.get("num_train", 100),
                    alpha=self.decoding_config.get("alpha", 0.5),
                )
                return (responses, [[] for _ in responses]) if collect_stats else responses

            if self.decoding_method == "cad":
                responses = cad_generation_batch(
                    model=self.base_model,
                    tokenizer=self.base_tokenizer,
                    input_ids_batch=base_inputs["input_ids"],
                    noisy_input_ids_batch=noisy_inputs["input_ids"],
                    eta=self.decoding_config.get("eta", 0.5),
                    max_new_tokens=self.max_new_tokens,
                )
                return (responses, [[] for _ in responses]) if collect_stats else responses

        if self.decoding_method == "knn_lm":
            cfg = self.decoding_config
            return knn_lm_decoding_batch(
                model=self.base_model,
                model_name=cfg.get("model_name", ""),
                tokenizer=self.base_tokenizer,
                input_ids_batch=base_inputs["input_ids"],
                train_data=cfg.get("train_data", ""),
                num_train=cfg.get("num_train", 100),
                max_new_tokens=self.max_new_tokens,
                alpha=cfg.get("alpha", 0.5),
                collect_stats=collect_stats,
            )

        if self.decoding_method == "rcd":
            cfg = self.decoding_config
            knn_ds = cfg.get("knn_datastore") or _load_knn_datastore(
                cfg.get("train_data", ""),
                cfg.get("model_name", ""),
                cfg.get("embed_model_name", _DEFAULT_EMBED_MODEL),
                use_gpu=torch.cuda.is_available(),
            )
            return rcd_generation_batch(
                model=self.base_model,
                model_name=cfg.get("model_name", ""),
                tokenizer=self.base_tokenizer,
                input_ids_batch=base_inputs["input_ids"],
                knn_datastore=knn_ds,
                max_new_tokens=self.max_new_tokens,
                alpha=cfg.get("alpha", 0.5),
                shaping_mode=cfg.get("shaping_mode", "linear"),
                top_n=cfg.get("top_n", None),
                sim_threshold=cfg.get("sim_threshold", 0.7),
                agg_mode=cfg.get("agg_mode", "weighted"),
                chunk_size=cfg.get("chunk_size", 8),
                exact_match=cfg.get("exact_match", ""),
                collect_stats=collect_stats,
            )

        # Fallback: HuggingFace .generate() (greedy, dola, icl, …)
        outputs = self.base_model.generate(
            **base_inputs,
            generation_config=self.generation_config,
            output_scores=True,
            return_dict_in_generate=True,
        )
        input_len = base_inputs["input_ids"].shape[-1]
        responses = self.base_tokenizer.batch_decode(
            outputs.sequences[:, input_len:], skip_special_tokens=True
        )
        if collect_stats and hasattr(outputs, "scores") and outputs.scores:
            batch_size = len(responses)
            hf_stats: list[list[dict]] = [[] for _ in range(batch_size)]
            for step_scores in outputs.scores:
                for b in range(batch_size):
                    hf_stats[b].append(
                        _compute_step_stats(step_scores[b], int(step_scores[b].argmax().item()))
                    )
            return responses, hf_stats
        return (responses, [[] for _ in responses]) if collect_stats else responses


# ---------------------------------------------------------------------------
# AllDecoding — high-level orchestrator
# ---------------------------------------------------------------------------

class AllDecoding:
    """
    High-level wrapper that handles prompt construction, retrieval (KAPING /
    kNN-ICL), and calls BaseGenerator for inference.
    """

    def __init__(
        self,
        model_path: str,
        decoding_method: str,
        max_new_tokens: Optional[int] = None,
        decoding_config: Optional[dict] = None,
        prompt_key: str = "zero_shot",
        noisy_prompt_key: Optional[str] = None,
        device: str = "auto",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        train_data: str = "wiki",
    ):
        decoding_config = decoding_config or {}
        generation_config: dict = {}

        if decoding_method == "dola":
            generation_config["do_layers"] = "high"
            generation_config.setdefault("generation_penalty", 1.2)
        elif decoding_method in {"greedy", "icl"}:
            generation_config = {
                "do_sample": False,
                "top_k": 0,
                "top_p": 1.0,
                "num_beams": 1,
            }

        if max_new_tokens is not None:
            generation_config["max_new_tokens"] = max_new_tokens

        self.model = BaseGenerator(
            model_path=model_path,
            decoding_method=decoding_method,
            generation_config=generation_config,
            decoding_config=decoding_config,
            device=device,
        )
        self.prompt_loader = PromptTemplateLoader()
        self.prompt_key = prompt_key
        self.noisy_prompt_key = noisy_prompt_key
        self.decoding_method = decoding_method
        self.decoding_config = decoding_config
        self.train_data = train_data

        # Embedding-based retrieval (KAPING / kNN-ICL)
        self.index = None
        self.triplets = None
        self.icl_index = None
        self.icl_data = None
        if decoding_method in {"KAPING", "kNN-ICL"}:
            from utils.baseline import load_triplets
            from utils.data import load_data

            embed_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedding_model = SentenceTransformer(embedding_model, device=embed_device)

            if decoding_method == "KAPING":
                self.triplets = load_triplets(train_data, num_train=100)
                triplet_texts = [
                    f"{t['head']} {t['type']} {t['tail']}" for t in self.triplets
                ]
                embs = self.embedding_model.encode(triplet_texts, convert_to_numpy=True)
                embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
                self.index = faiss.IndexFlatIP(embs.shape[1])
                self.index.add(embs)
            else:
                # kNN-ICL: separate ICL index over training questions
                self.icl_data = load_data(train_data=train_data, num_train=100)
                questions  = [d["question"] for d in self.icl_data]
                q_embs = self.embedding_model.encode(questions, convert_to_numpy=True)
                q_embs = q_embs / np.linalg.norm(q_embs, axis=1, keepdims=True)
                self.icl_index = faiss.IndexFlatIP(q_embs.shape[1])
                self.icl_index.add(q_embs)

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def get_top_k_triplets(self, question: str, k: int = 3) -> list[dict]:
        """Retrieve top-k KAPING knowledge triplets for a question."""
        q_emb = self.embedding_model.encode([question], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        _, idxs = self.index.search(q_emb, k)
        return [self.triplets[i] for i in idxs[0]]

    def get_top_k_examples(self, question: str, k: int = 3) -> list[dict]:
        """Retrieve top-k similar training examples for kNN-ICL."""
        q_emb = self.embedding_model.encode([question], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        _, idxs = self.icl_index.search(q_emb, k)
        return [self.icl_data[i] for i in idxs[0]]

    def _format_kaping_facts(self, retrieved: list[dict]) -> str:
        return "".join(f"- ({t['head']}, {t['type']}, {t['tail']})\n" for t in retrieved)

    def _format_icl_examples(self, retrieved: list[dict]) -> str:
        lines = []
        for item in retrieved:
            ans = item.get("correct_answers") or item.get("answer", "")
            if isinstance(ans, list):
                ans = ans[0] if ans else ""
            lines.append(f"Q: {item['question']}\nA: {ans}\n")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inference_on_dataset(
        self,
        questions: list[str],
        answers: Optional[list[str]] = None,
        batch_size: int = 16,
        stats_path: Optional[str] = None,
    ) -> list[str]:
        prompts = self._build_prompts(questions, answers)
        noisy_prompts = self._build_noisy_prompts(questions)
        return self.model.inference_on_data(
            prompts, batch_size=batch_size, noisy_prompts=noisy_prompts, stats_path=stats_path
        )

    def generate(self, prompt: str, noisy_prompt: Optional[str] = None) -> str:
        return self.model.inference_one_sample(prompt, noisy_prompt)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompts(
        self,
        questions: list[str],
        answers: Optional[list[str]] = None,
    ) -> list[str]:
        prompts = []
        for i, question in enumerate(questions):
            if self.decoding_method == "KAPING":
                retrieved = self.get_top_k_triplets(question)
                placeholders = {"question": question, "facts": self._format_kaping_facts(retrieved)}
                prompt_group = "few_shot"
                template_name = "default"
            elif self.decoding_method == "kNN-ICL":
                retrieved = self.get_top_k_examples(question)
                examples  = self._format_icl_examples(retrieved)
                prompt_group  = "few_shot_bio" if self.train_data == "bio" else "few_shot_icl"
                placeholders  = {"question": question, "examples": examples}
                template_name = "default"
            else:
                prompt_group  = self.prompt_key
                placeholders  = {"question": question}
                if answers is not None and i < len(answers):
                    placeholders["answer"] = answers[i]
                template_name = "cot" if self.decoding_method == "dola" else "default"

            chat_input = self.prompt_loader.construct_chat_input(
                prompt_group=prompt_group,
                template_name=template_name,
                placeholders=placeholders,
                tokenizer=self.model.base_tokenizer,
            )
            prompts.append(chat_input)

        return prompts

    def _build_noisy_prompts(self, questions: list[str]) -> Optional[list[str]]:
        if self.decoding_method not in _DECODING_METHODS_NEEDING_NOISY:
            return None
        if self.noisy_prompt_key is None:
            raise ValueError(
                f"noisy_prompt_key must be set for decoding method '{self.decoding_method}'."
            )
        noisy_prompts = []
        for question in questions:
            noisy_prompts.append(
                self.prompt_loader.construct_chat_input(
                    prompt_group=self.noisy_prompt_key,
                    placeholders={"question": question},
                    tokenizer=self.model.base_tokenizer,
                )
            )
        return noisy_prompts