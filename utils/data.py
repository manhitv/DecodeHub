"""
utils/data.py — Dataset loaders and text extraction utilities.

All public loaders return a list of dicts with at least:
    {"question": str, "correct_answers": list[str]}
and optionally "incorrect_answers", "knowledge", etc.
"""

from __future__ import annotations

import json
import random
import re
import string
from copy import deepcopy
from itertools import product          # kept for downstream callers
from pathlib import Path
from typing import Optional

import jsonlines
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils.config import data_dir, precisewiki_raw

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

_data = Path(data_dir)
BIOGRAPHY_DATA_PATH   = _data / "article_200.json"
PRECISEWIKI_FILE_PATH = _data / "precisewiki_{split}.jsonl"
PRECISEWIKI_RAW_PATH  = Path(precisewiki_raw) if precisewiki_raw else Path("")

_HALU_DATASETS = {"halu_qa", "halu_dia", "halu_sum"}
_FAITH_DATASETS = {"unanswerable", "inconsistent", "counterfactual"}


# ===========================================================================
# Individual dataset loaders
# ===========================================================================

def load_truthfulqa(split: str, max_sample_num: int = 817) -> list[dict]:
    try:
        ori = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
    except ValueError:
        # datasets>=4.0 removed the 'List' feature type (renamed to 'Sequence').
        # The cached dataset_info.json may still use the old type; delete the cache
        # and re-download so it is rebuilt with the current schema.
        import shutil
        cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "truthfulqa___truthful_qa"
        if cache_root.exists():
            shutil.rmtree(cache_root)
        ori = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
    dataset = [
        {
            "question": s["question"],
            "correct_answers": [s["best_answer"]] + s["correct_answers"],
            "incorrect_answers": s["incorrect_answers"],
        }
        for s in ori
    ]
    slices = {"validation": slice(None, 400), "test": slice(400, 817)}
    if split not in slices:
        raise ValueError(f"Split '{split}' not supported for TruthfulQA.")
    dataset = dataset[slices[split]]
    return dataset[:max_sample_num]


def load_wiki(split: str, max_sample_num: int = 100) -> list[dict]:
    _split_map = {"test": "test", "train": "train"}
    if split not in _split_map:
        raise ValueError(f"Split '{split}' not supported for WikiQA.")
    ds = load_dataset("microsoft/wiki_qa")[_split_map[split]]

    from collections import OrderedDict
    groups: dict = OrderedDict()
    for s in ds:
        qid = s["question_id"]
        if qid not in groups:
            groups[qid] = {"question": s["question"], "correct_answers": [], "incorrect_answers": []}
        if s["label"] == 1:
            groups[qid]["correct_answers"].append(s["answer"])
        else:
            groups[qid]["incorrect_answers"].append(s["answer"])
    return list(groups.values())[:max_sample_num]


def load_alpaca(split: str, max_sample_num: int = 100) -> list[dict]:
    if split == "test":
        ds = load_dataset(
            "json",
            data_files={"eval": "https://huggingface.co/datasets/tatsu-lab/alpaca_eval"
                                 "/resolve/main/alpaca_eval.json"},
        )["eval"]
    elif split == "train":
        ds = load_dataset("tatsu-lab/alpaca")["train"]
    else:
        raise ValueError(f"Split '{split}' not supported for Alpaca.")
    dataset = [
        {"question": s["instruction"], "correct_answers": [s["output"]]}
        for s in ds
    ]
    return dataset[:max_sample_num]


def load_gsm8k(split: str, max_sample_num: int = 100) -> list[dict]:
    _split_map = {"test": "test", "train": "train"}
    if split not in _split_map:
        raise ValueError(f"Split '{split}' not supported for GSM8K.")
    ds = load_dataset("openai/gsm8k", "main")[_split_map[split]]
    dataset = [
        {"question": s["question"], "correct_answers": [s["answer"]]}
        for s in ds
    ]
    return dataset[:max_sample_num]


def load_biography(split: str, max_sample_num: int = 128) -> list[dict]:
    with BIOGRAPHY_DATA_PATH.open() as f:
        data = json.load(f)
    dataset = [{"question": name, "answer": bio} for name, bio in data.items()]
    slices = {"test": slice(100, None), "validation": slice(None, 100)}
    if split not in slices:
        raise ValueError(f"Split '{split}' not supported for Biography.")
    return dataset[slices[split]][:max_sample_num]


# ------------------------------------------------------------------
# HaluEval
# ------------------------------------------------------------------

def _load_halu(
    halu_key: str,   # "qa" | "dialogue" | "summarization"
    split: str,
    max_sample_num: int,
    question_field: str,
    correct_field: str,
    incorrect_field: str,
    extra_fields: Optional[list[str]] = None,
    doc_filter=None,
) -> list[dict]:
    ds = load_dataset("pminervini/HaluEval", halu_key)["data"]
    dataset = []
    for s in ds:
        if doc_filter and not doc_filter(s):
            continue
        entry = {
            "question":          s[question_field],
            "correct_answers":   s[correct_field],
            "incorrect_answers": s[incorrect_field],
        }
        for f in (extra_fields or []):
            entry[f] = s[f]
        dataset.append(entry)

    slices = {"test": slice(100, None), "train": slice(None, 100)}
    if split not in slices:
        raise ValueError(f"Split '{split}' not supported for HaluEval.")
    return dataset[slices[split]][:max_sample_num]


def load_halu_qa(split: str, max_sample_num: int = 100) -> list[dict]:
    return _load_halu(
        "qa", split, max_sample_num,
        question_field="question",
        correct_field="right_answer",
        incorrect_field="hallucinated_answer",
        extra_fields=["knowledge"],
    )


def load_halu_dia(split: str, max_sample_num: int = 100) -> list[dict]:
    return _load_halu(
        "dialogue", split, max_sample_num,
        question_field="dialogue_history",
        correct_field="right_response",
        incorrect_field="hallucinated_response",
        extra_fields=["knowledge"],
    )


def load_halu_sum(split: str, max_sample_num: int = 100) -> list[dict]:
    return _load_halu(
        "summarization", split, max_sample_num,
        question_field="document",
        correct_field="right_summary",
        incorrect_field="hallucinated_summary",
        doc_filter=lambda s: len(s["document"]) <= 1500,
    )


# ------------------------------------------------------------------
# FaithEval
# ------------------------------------------------------------------

def load_faith(
    split: str,
    faith_data: str,
    max_sample_num: int = 300,
) -> list[dict]:
    if faith_data not in _FAITH_DATASETS:
        raise ValueError(f"faith_data must be one of {_FAITH_DATASETS}.")

    ori = load_dataset(f"Salesforce/FaithEval-{faith_data}-v1.0", split="test")
    dataset = []
    for s in ori:
        if faith_data in ("unanswerable", "inconsistent"):
            dataset.append({
                "question":        f"Context: {s['context']}\nQuestion: {s['question']}",
                "correct_answers": s["answers"],
            })
        else:  # counterfactual
            choices = "\n".join(
                f"{lbl}. {txt}"
                for lbl, txt in zip(s["choices"]["label"], s["choices"]["text"])
            )
            dataset.append({
                "question":        (
                    f"Context: {s['context']}\nQuestion: {s['question']}\nChoices: {choices}"
                ),
                "correct_answers": [s["answer"]],
                "answer_key":      s["answerKey"],
            })

    slices = {"train": slice(None, 200), "validation": slice(100, 200), "test": slice(200, None)}
    if split not in slices:
        raise ValueError(f"Split '{split}' not supported for FaithEval.")
    return dataset[slices[split]][:max_sample_num]


# ------------------------------------------------------------------
# PreciseWiki
# ------------------------------------------------------------------

def load_precisewiki(split: str, max_sample_num: int = 100) -> list[dict]:
    path = Path(str(PRECISEWIKI_FILE_PATH).format(split=split))
    dataset = []
    with jsonlines.open(path) as reader:
        for s in reader:
            dataset.append({"question": s["question"], "correct_answers": [s["answer"]]})
            if len(dataset) >= max_sample_num:
                break
    return dataset


def pre_load_precisewiki(split: str, max_sample_num: int = 100) -> list[dict]:
    if not PRECISEWIKI_RAW_PATH or not PRECISEWIKI_RAW_PATH.exists():
        raise FileNotFoundError(
            "precisewiki_raw is not configured. Set paths.precisewiki_raw in config.yaml "
            "or the DECODEHUB_PRECISEWIKI_RAW environment variable."
        )
    n_per_bin = max_sample_num // 10
    with PRECISEWIKI_RAW_PATH.open(encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    offsets = {"test": (100, n_per_bin + 100), "train": (0, n_per_bin)}
    if split not in offsets:
        raise ValueError(f"Split '{split}' not supported for pre_load_precisewiki.")
    lo, hi = offsets[split]

    dataset = []
    for cat in range(10):
        indices = [j for j, d in enumerate(data) if d["h_score_cat"] == cat][lo:hi]
        dataset.extend(
            {k: data[j][k] for k in ("pageid", "title", "document", "h_score_cat")}
            for j in indices
        )
    return dataset


# ===========================================================================
# Unified loader
# ===========================================================================

_LOADERS: dict = {
    "truthful_qa": lambda split, n: load_truthfulqa(split, n),
    "gsm8k":       lambda split, n: load_gsm8k(split, n),
    "bio":         lambda split, n: load_biography("validation", n),
    "wiki":        lambda split, n: load_wiki(split, n),
    "precisewiki": lambda split, n: load_precisewiki(split, n),
    "alpaca":      lambda split, n: load_alpaca(split, n),
    "halu_qa":     lambda split, n: load_halu_qa(split, n),
    "halu_dia":    lambda split, n: load_halu_dia(split, n),
    "halu_sum":    lambda split, n: load_halu_sum(split, n),
}


def _resolve_loader(train_data: str):
    if train_data in _LOADERS:
        return _LOADERS[train_data]
    if train_data.startswith("faith_"):
        faith_key = train_data.split("_", 1)[1]
        return lambda split, n: load_faith(split, faith_key, n)
    raise ValueError(f"Dataset '{train_data}' is not supported.")


def load_data(train_data: str, num_train: int = 100) -> list[dict]:
    """Return up to *num_train* samples from the training split of *train_data*."""
    split = "validation" if train_data in ("truthful_qa", "bio") else "train"
    return _resolve_loader(train_data)(split, num_train)


# ===========================================================================
# Text extraction for kNN-LM / RAD datastore construction
# ===========================================================================

def extract_correct_answers(item: dict, train_data: str) -> list[str]:
    """Return a list of correct-answer strings for *item* from *train_data*.

    Handles the key differences across datasets:
    - bio:      answer is under "answer" (string)
    - halu_*:   correct_answers is a single string, not a list
    - others:   correct_answers is already a list
    """
    if train_data == "bio":
        return [item["answer"]]
    if train_data in ("halu_qa", "halu_dia", "halu_sum"):
        return [item["correct_answers"]]   # single string in HaluEval
    return list(item["correct_answers"])


def load_texts(
    train_data: str,
    num_train: int,
    noisy_level: int = 0,
) -> list[str]:
    """Return question+answer concatenations for all training samples."""
    ds = load_data(train_data, num_train)

    if noisy_level > 0:
        ds = inject_noise(ds, noisy_level=noisy_level, seed=42)

    texts = []
    for item in tqdm(ds[:num_train], desc="Extracting texts"):
        question = item["question"]
        if train_data == "faith_counterfactual":
            question = question.split("\nChoices:")[0]
        for answer in extract_correct_answers(item, train_data):
            texts.append(f"{question} {answer}")
    return texts


# ===========================================================================
# Noise injection
# ===========================================================================

def inject_noise(
    dataset: list[dict],
    noisy_level: int,
    seed: int = 42,
) -> list[dict]:
    """
    Replace correct_answers of ``noisy_level``% of samples with answers from
    other randomly-chosen samples (simulates conflicting grounding).
    """
    if noisy_level == 0:
        return dataset

    random.seed(seed)
    dataset = deepcopy(dataset)

    n = len(dataset)
    num_noisy = int(n * noisy_level / 100)
    noisy_indices = random.sample(range(n), num_noisy)

    # bio uses "answer"; all other datasets use "correct_answers"
    answer_key = "correct_answers" if "correct_answers" in dataset[0] else "answer"
    all_answers = [s[answer_key] for s in dataset]
    shuffled = all_answers.copy()
    random.shuffle(shuffled)

    for i, idx in enumerate(noisy_indices):
        dataset[idx][answer_key] = shuffled[i]
    return dataset


# ===========================================================================
# PreciseWiki: question/answer generation helpers
# ===========================================================================

PRECISE_Q_GENERATION_PROMPT = """\
I would like you to act as a question generator. I will provide reference and you \
will generate a factual knowledge based question about "{wiki_title}" based on the \
reference. The specific requirements are as follows:

1. The question can be fully answered based only on the reference material.
2. The question should be objective and not open-ended.
3. The question should be concise.
4. The question should not require additional information to answer.
5. The question's answer should be a word or a phrase.
6. The question should have only one answer.

Reference:
{wiki_document}

Please reply with the question only without any explanation or additional information:
"""

PRECISE_ANSWERABILITY_PROMPT = """\
I would like you to judge question's answerability and answer the question.
I will provide a question and reference document, and you will judge whether the \
question is fully answerable based only on the reference document, i.e., whether \
the answer is included in the reference.
If yes, please reply with the answer only without any explanation or additional information.
If no, please reply with "unanswerable" only.

Reference document: {ref_document}

Question: {question}\
"""


def _justify_answerability(reply: str) -> str | int:
    lowered = reply.strip().lower()
    if (
        lowered == "unanswerable"
        or "unanswerable" in lowered
        or lowered.startswith("unfortunately")
        or len(reply.split()) > 20
    ):
        return -1
    return reply.strip()


def _cohere_batch(message_list: list, cohere_client) -> list[str]:
    responses = []
    for messages in message_list:
        response = cohere_client.chat(messages=messages, model=cohere_client.model)
        responses.append(response.message.content[0].text)
    return responses


def prepare_precisewiki(
    split: str,
    llm_client,
    max_sample_num: int = 100,
) -> list[dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    raw = pre_load_precisewiki(split=split, max_sample_num=max_sample_num)
    all_data, q_prompts = [], []

    for item in raw:
        sections = split_doc(item["document"], "en", encoder, keep_end=False, keep_colon=False)
        if len(sections) > 2:
            sections = sections[:-1]
        section = random.choice(sections)

        obj = {k: item[k] for k in ("title", "h_score_cat", "pageid")}
        obj["reference"] = section
        all_data.append(obj)

        q_prompts.append([{
            "role": "user",
            "content": PRECISE_Q_GENERATION_PROMPT.format(
                wiki_title=item["title"], wiki_document=section.strip()
            ),
        }])

    questions = _cohere_batch(q_prompts, llm_client)

    a_prompts = []
    for i, question in enumerate(questions):
        prompt = PRECISE_ANSWERABILITY_PROMPT.format(
            ref_document=all_data[i]["reference"], question=question.strip()
        )
        all_data[i]["question"] = prompt
        a_prompts.append([{"role": "user", "content": prompt}])

    answers = _cohere_batch(a_prompts, llm_client)

    output_path = Path(f"data/precisewiki_{split}.jsonl")
    filter_count = 0
    for i, answer in enumerate(answers):
        justified = _justify_answerability(answer)
        if justified == -1:
            filter_count += 1
            continue
        all_data[i]["answer"] = answer
        with jsonlines.open(output_path, mode="a") as writer:
            writer.write(all_data[i])

    print(f"Done. Filtered {filter_count} unanswerable questions.")
    return all_data


# ===========================================================================
# Text splitting utilities (document chunking for kNN-LM / embeddings)
# ===========================================================================

def split_doc(
    text: str,
    language: str,
    encoding,
    keep_end: bool,
    keep_colon: bool,
    min_len: Optional[int] = None,
    max_len: Optional[int] = None,
) -> list[str]:
    defaults = {"zh": (108, 270), "en": (80, 200)}
    if min_len is None or max_len is None:
        min_len, max_len = defaults[language]

    if len(encoding.encode(text)) <= max_len:
        return [text]

    if language == "zh":
        pattern = r"(##+)|(\n(?:\d+|[一二三四五六七八九十⓪①②③④⑤⑥⑦⑧⑨⑩零壹贰叁肆伍陆柒捌玖拾])[、.：:].{0,6}\n+)"
    else:
        pattern = r"(##+)|(\n(?:\d+|[⓪①②③④⑤⑥⑦⑧⑨⑩])[、.：:].{0,15}\n+)"

    parts = re.split(pattern, text)
    sections = [parts[0]] if parts[0].strip() else []
    assert (len(parts) - 1) % 3 == 0
    for i in range(1, len(parts), 3):
        seg = "".join(p for p in parts[i : i + 3] if p)
        if seg:
            sections.append(seg)

    chunks = []
    for section in sections:
        chunks += _split_context(section, max_len, language, encoding, keep_end, keep_colon)

    if not chunks:
        return [text]

    # Merge short trailing chunks
    merged, buf = [], chunks[0]
    for chunk in chunks[1:]:
        if len(encoding.encode(chunk)) < min_len:
            buf = (buf + chunk) if language == "zh" else (buf + " " + chunk)
        else:
            if buf.strip():
                merged.append(buf)
            buf = chunk
    if buf.strip():
        merged.append(buf)
    return merged


def _split_context(
    text: str,
    max_len: int,
    language: str,
    encoding,
    keep_end: bool,
    keep_colon: bool,
) -> list[str]:
    min_len = 14 if language == "zh" else 12
    if len(encoding.encode(text)) <= max_len:
        return [text]

    sents = _sentence_tokenize(text, language, keep_end, keep_colon)
    buf = sents[0]
    chunks = []

    def _join(a: str, b: str) -> str:
        if keep_end or language == "zh" and re.search(r"\W$", a):
            return a + b
        return a + " " + b

    for sent in sents[1:-1]:
        if len(encoding.encode(buf)) + len(encoding.encode(sent)) > max_len:
            chunks.append(buf)
            buf = sent
        else:
            buf = _join(buf, sent)

    last = sents[-1]
    if len(encoding.encode(last)) <= min_len or (
        len(encoding.encode(sents[-2] if len(sents) > 1 else "")) + len(encoding.encode(last)) <= max_len
    ):
        buf = _join(buf, last)
    else:
        chunks.append(buf)
        buf = last

    chunks.append(buf)
    return chunks


# -- Sentence tokenization helpers -----------------------------------------

_DOT_ABBREVS = (
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec|No|Op|D|Dr|St)"
)


def _normalize_dots(text: str) -> str:
    """Collapse abbreviation dots so sentence splitters don't break on them."""
    text = re.sub(r"O\.S\.B\.M\. ", "O.S.B.M.", text)
    text = re.sub(r"(\W|^)([A-Z]\.) ?([A-Z]\.) ?([A-Za-z])", r"\1\2\3\4", text)
    text = re.sub(r"(\W|^)([A-Z]\.) ?([A-Za-z])", r"\1\2\3", text)
    text = re.sub(r"((\n\s*)|(\. ))(\d+)\.\s+", r"\1\4.", text)
    text = re.sub(r"^(\d+)\.\s+", r"\1.", text)
    text = re.sub(rf"(\W|^){_DOT_ABBREVS}\.\s+", r"\1\2.", text)
    text = re.sub(r"(\W|^)(et al)\.\s+([a-z])", r"\1\2.\3", text)
    text = re.sub(r"Alexander v\. Holmes", "Alexander v.Holmes", text)
    text = re.sub(r"Brown v\. Board", "Brown v.Board", text)
    return text


def _restore_dots(text: str) -> str:
    text = re.sub(r"^(\d+)\.", r"\1. ", text)
    text = re.sub(r"(\W|^)([A-Z]\.) ?([A-Z]\.) ?([A-Za-z])", r"\1\2 \3 \4", text)
    text = re.sub(r"(\W|^)([A-Z]\.) ?([A-Z][a-z])", r"\1\2 \3", text)
    text = re.sub(rf"(\W|^){_DOT_ABBREVS}\.", r"\1\2. ", text)
    text = re.sub(r"(\W|^)(et al)\.([a-z])", r"\1\2. \3", text)
    for pattern, repl in [
        (r"O\.S\.B\.M\.", "O.S.B.M. "), (r"U\. +S\.", "U.S."),
        (r"U\.S\. *S\. *R\.", "U.S.S.R."), (r"D\. +C\.", "D.C."),
        (r"D\. +Roosevelt", "D. Roosevelt"), (r"A\. *D\. *(\W)", r"A.D.\1"),
        (r"A\. +D\.", "A.D."), (r"F\. +C\.", "F.C."), (r"J\. +League", "J.League"),
        (r"Alexander v\. *Holmes", "Alexander v. Holmes"),
        (r"Brown v\. *Board", "Brown v. Board"),
    ]:
        text = re.sub(pattern, repl, text)
    return text


def _sentence_tokenize(
    text: str,
    language: str,
    keep_end: bool,
    keep_colon: bool,
) -> list[str]:
    if language == "zh":
        if not keep_colon:
            text = re.sub(r"([:：])(\s+)", r"\1", text)
        raw = re.split(r"(。|！|？|；|\n+)", text)
    else:
        text = _normalize_dots(text)
        if not keep_colon:
            text = re.sub(r"([:：])(\s+)", r"\1 ", text)
        raw = re.split(r"((?:[.!?;]\s+)|(?:\n+))", text)

    sents = []
    for i in range(0, len(raw), 2):
        seg = raw[i] + (raw[i + 1] if i + 1 < len(raw) else "")
        if not keep_end:
            seg = seg.strip()
        if seg:
            if language == "en":
                seg = _restore_dots(seg)
            sents.append(seg)
    return sents


# ===========================================================================
# Miscellaneous helpers (kept for downstream compatibility)
# ===========================================================================

def faith_normalize_answer(s: str) -> str:
    def remove_articles(t):  return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t):  return " ".join(t.split())
    def handle_punc(t):
        exclude = set(string.punctuation + "''´`")
        return "".join(c if c not in exclude else " " for c in t)
    def replace_underscore(t): return t.replace("_", " ")
    return white_space_fix(remove_articles(handle_punc(s.lower().replace("_", " ")))).strip()


def parse_bullets(sentence: str) -> list[str]:
    sentence = sentence.replace("*", "")
    for marker in ("Question 2", None):
        if marker and sentence.find(marker) != -1:
            sentence = sentence[sentence.find(marker):]
    low = sentence.lower()
    if "refined answer" in low:
        idx = low.find("refined answer") + len("refined answer")
        sentence = sentence[idx:]
        sentence = sentence[sentence.find("\n") + 1:]

    lines = [l for l in sentence.split("\n") if l]
    if len(lines) == 1:
        lines = [l + "." for l in sentence.split(".") if l]

    skip_phrases = ("Here is", "Here are", "I apologize", "Sorry", "Thank you", "Please", "I hope")
    lines = [l for l in lines if not any(p in l for p in skip_phrases) and l[-1] != ":"]

    bullets = []
    for line in lines:
        try:
            idx = line.find(next(filter(str.isalpha, line)))
        except StopIteration:
            continue
        if line[idx:]:
            bullets.append(line[idx:])
    return bullets


def parse_yes_no(s: str) -> Optional[bool]:
    low = s.lower()
    if "yes" in low:
        return True
    if "no" in low:
        return False
    return None


def filter_people(person: str) -> str:
    return person.split("(")[0]


# ===========================================================================
# CLI entry point — prepare PreciseWiki dataset
# ===========================================================================

if __name__ == "__main__":
    import argparse
    import cohere
    from utils import api_key

    parser = argparse.ArgumentParser(description="Prepare PreciseWiki dataset splits.")
    parser.add_argument("--split",          type=str, default="test",
                        help="Dataset split to process (e.g. 'test', 'train').")
    parser.add_argument("--max_sample_num", type=int, default=100,
                        help="Maximum number of samples to process.")
    args = parser.parse_args()

    client = cohere.ClientV2(api_key=api_key.cohere_api_key)
    print(f"Preparing PreciseWiki split='{args.split}', max_sample_num={args.max_sample_num}")
    data = prepare_precisewiki(
        split=args.split,
        llm_client=client,
        max_sample_num=args.max_sample_num,
    )
    print(f"Done. Processed {len(data)} samples.")