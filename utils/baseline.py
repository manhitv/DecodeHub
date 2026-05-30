"""
utils/baseline.py — ICL baseline utilities for KAPING and kNN-ICL.

KAPING: extracts knowledge-graph triplets via REBEL, embeds them, retrieves
        the top-k most relevant triplets at inference time.

kNN-ICL: embeds training questions, retrieves the top-k most similar Q&A
         pairs at inference time to use as few-shot context.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils.config import data_dir
from utils.data import load_data, load_texts

_REBEL_MODEL = "Babelscape/rebel-large"
_DEFAULT_EMBED = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# REBEL — knowledge triplet extraction
# ---------------------------------------------------------------------------

def extract_triplets(generated_text: str) -> list[dict]:
    """Parse REBEL output into a list of {'head', 'type', 'tail'} dicts."""
    triplets = []
    subject = relation = object_ = ""
    current = "x"
    for token in (
        generated_text
        .replace("<s>", "").replace("<pad>", "").replace("</s>", "")
        .split()
    ):
        if token == "<triplet>":
            if relation:
                triplets.append({"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()})
                relation = ""
            subject, current = "", "t"
        elif token == "<subj>":
            if relation:
                triplets.append({"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()})
            object_, current = "", "s"
        elif token == "<obj>":
            relation, current = "", "o"
        else:
            if current == "t":
                subject  += " " + token
            elif current == "s":
                object_  += " " + token
            elif current == "o":
                relation += " " + token

    if subject and relation and object_:
        triplets.append({"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()})
    return triplets


def build_knowledge_graph(
    train_data: str,
    num_train:  int  = 100,
    save_dir:   Optional[str] = None,
    batch_size: int  = 32,
) -> list[dict]:
    """Run REBEL on training texts and save the triplet corpus to disk.

    Output file: {save_dir}/{train_data}_{num_train}_triplets.json
    Returns the list of triplet dicts.
    """
    save_dir = save_dir or data_dir
    out_path = Path(save_dir) / f"{train_data}_{num_train}_triplets.json"

    tokenizer = AutoTokenizer.from_pretrained(_REBEL_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(_REBEL_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    texts = load_texts(train_data=train_data, num_train=num_train)
    triplet_corpus: list[dict] = []

    for batch_start in tqdm(range(0, len(texts), batch_size), desc="REBEL extraction"):
        batch = texts[batch_start : batch_start + batch_size]
        enc   = tokenizer(batch, max_length=256, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(
                enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
                max_length=256,
                length_penalty=1,
                num_beams=3,
                num_return_sequences=1,
            )
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=False)
        for idx, text in enumerate(decoded):
            global_idx = batch_start + idx
            for triplet in extract_triplets(text):
                triplet["source_idx"] = global_idx
                triplet_corpus.append(triplet)

    os.makedirs(save_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(triplet_corpus, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(triplet_corpus)} triplets → {out_path}")
    return triplet_corpus


def load_triplets(
    train_data: str,
    num_train:  int = 100,
    save_dir:   Optional[str] = None,
) -> list[dict]:
    """Load a pre-built triplet corpus from disk."""
    save_dir = save_dir or data_dir
    path = Path(save_dir) / f"{train_data}_{num_train}_triplets.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Triplet corpus not found at {path}. "
            f"Run build_knowledge_graph('{train_data}', {num_train}) first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# kNN-ICL index — embed training questions
# ---------------------------------------------------------------------------

def build_icl_index(
    train_data:       str,
    num_train:        int = 100,
    embed_model_name: str = _DEFAULT_EMBED,
    save_dir:         Optional[str] = None,
) -> tuple[faiss.Index, list[dict]]:
    """Embed training questions and return a FAISS cosine-similarity index.

    Returns (faiss_index, train_items).
    """
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    embed_mdl  = SentenceTransformer(embed_model_name, device=device)
    train_data_list = load_data(train_data=train_data, num_train=num_train)

    questions  = [d["question"] for d in train_data_list]
    embs       = embed_mdl.encode(questions, convert_to_numpy=True)
    embs       = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    embs       = embs.astype("float32")

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index, train_data_list


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def retrieve_icl_examples(
    question:    str,
    embed_model: SentenceTransformer,
    icl_index:   faiss.Index,
    train_items: list[dict],
    k:           int = 3,
) -> list[dict]:
    """Return the top-k training items most similar to *question*."""
    emb = embed_model.encode([question], convert_to_numpy=True)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    _, idxs = icl_index.search(emb.astype("float32"), k)
    return [train_items[i] for i in idxs[0]]


def format_icl_examples(examples: list[dict], train_data: str) -> str:
    """Format retrieved examples as a Q&A few-shot block."""
    lines = []
    for item in examples:
        q = item["question"]
        if train_data == "bio":
            a = item.get("answer", "")
        else:
            ans = item.get("correct_answers") or item.get("answer", "")
            a   = ans[0] if isinstance(ans, list) and ans else str(ans)
        lines.append(f"Q: {q}\nA: {a}\n")
    return "\n".join(lines)


def retrieve_kaping_facts(
    question:     str,
    embed_model:  SentenceTransformer,
    triplet_index: faiss.Index,
    triplets:     list[dict],
    k:            int = 5,
) -> str:
    """Return a formatted bullet list of top-k KAPING triplets for *question*."""
    emb = embed_model.encode([question], convert_to_numpy=True)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    _, idxs = triplet_index.search(emb.astype("float32"), k)
    return "".join(
        f"- ({triplets[i]['head']}, {triplets[i]['type']}, {triplets[i]['tail']})\n"
        for i in idxs[0]
    )


# ---------------------------------------------------------------------------
# CLI — build a triplet corpus from the command line
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a KAPING knowledge-graph triplet corpus.")
    parser.add_argument("--train_data", type=str, default="truthful_qa")
    parser.add_argument("--num_train",  type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_dir",   type=str, default=None,
                        help="Directory for output JSON (defaults to config data_dir).")
    args = parser.parse_args()

    build_knowledge_graph(
        train_data = args.train_data,
        num_train  = args.num_train,
        save_dir   = args.save_dir,
        batch_size = args.batch_size,
    )
