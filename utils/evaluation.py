"""
utils/evaluation.py — Evaluation functions for all supported datasets.

Supports both LLM-based evaluation (Cohere, Gemini) and rule-based / metric-based
evaluation (ROUGE-L, BERTScore, FaithEval, HHEM).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import time
from typing import Optional

import numpy as np
import torch
from rouge_score import rouge_scorer
# NOTE: `bert_score` is imported lazily inside bertscore_f1() — it is only needed by
# the `open` eval metric. Importing it at module top-level previously made run.py
# (which imports this module for load_data_eval) fail under --run_only on any env
# without bert_score installed.

from utils.config import result_dir
from utils.data import (
    faith_normalize_answer,
    filter_people,
    load_alpaca,
    load_biography,
    load_faith,
    load_halu_dia,
    load_halu_qa,
    load_halu_sum,
    load_precisewiki,
    load_truthfulqa,
    load_wiki,
    parse_bullets,
    parse_yes_no,
)
from utils.prompt import PromptTemplateLoader

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
prompt_loader = PromptTemplateLoader()

# ---------------------------------------------------------------------------
# LLM evaluation back-ends
# ---------------------------------------------------------------------------

_COHERE_AUTH_KEYWORDS = ("bearer", "401", "unauthorized", "api key", "invalid token", "forbidden")
_COHERE_MAX_RETRIES   = 5


def _cohere_batch(message_list: list[list[dict]], client) -> list[str]:
    """Call Cohere chat API for each message list; return text responses.

    Retries on transient errors with exponential backoff (1-16 s).
    Auth errors (BearerToken / 401) are logged and not retried.
    """
    import time
    from utils import api_key as _api_key

    responses = []
    for messages in message_list:
        clean = [{"role": "user", "content": m["content"]} for m in messages]
        last_exc = None
        for attempt in range(_COHERE_MAX_RETRIES):
            try:
                resp = client.chat(messages=clean, model=_api_key.cohere_model)
                responses.append(resp.message.content[0].text)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                exc_lower = str(exc).lower()
                if any(kw in exc_lower for kw in _COHERE_AUTH_KEYWORDS):
                    print(f"[Cohere] Auth error — check your API key (not retrying): {exc}")
                    break
                wait = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
                print(f"[Cohere] Transient error (attempt {attempt + 1}/{_COHERE_MAX_RETRIES}, "
                      f"retry in {wait}s): {exc}")
                time.sleep(wait)
        if last_exc is not None:
            responses.append("")
    return responses


def _gemini_batch(message_list: list[list[dict]], client) -> list[str]:
    """Call Gemini generate_content API for each message list."""
    from utils import api_key as _api_key
    responses = []
    for messages in message_list:
        content = messages[0]["content"] if messages else ""
        resp = client.models.generate_content(model=_api_key.gemini_model, contents=content)
        responses.append(resp.text)
    return responses


def _llm_batch(message_list: list[list[dict]], args) -> list[str]:
    """Dispatch to the correct LLM back-end based on *args.evaluation_type*."""
    if args.evaluation_type == "cohere":
        return _cohere_batch(message_list, args.cohere_client)
    if args.evaluation_type == "gemini":
        return _gemini_batch(message_list, args.gemini_client)
    raise ValueError(f"Unsupported evaluation_type: '{args.evaluation_type}'")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _strip_response(text: str) -> str:
    """Remove trailing boilerplate added by some decoding methods."""
    text = text[text.rfind("\n") + 1:]
    if text.lower().startswith("**refined answer:**"):
        text = text[len("**Refined Answer:**"):]
    return text.strip()


def _first_alpha_word(text: str) -> str:
    """Return *text* with leading non-alpha characters removed."""
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[i:]
    return text


def _save_results(results: dict, file_name: str) -> None:
    os.makedirs(result_dir, exist_ok=True)
    with open(f"{result_dir}/{file_name}", "w") as f:
        json.dump(results, f, indent=4)


# ---------------------------------------------------------------------------
# Pretrained model evaluation (HHEM)
# ---------------------------------------------------------------------------

def eval_hhem(data: list[dict]) -> list[list[float]]:
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        "vectara/hallucination_evaluation_model",
        trust_remote_code=True,
        torch_dtype="auto",
    )
    scores = []
    for sample in data:
        response = _strip_response(sample["generated_answer"])
        refs = list(sample["correct_answers"])
        pair_scores = []
        for ref in refs:
            with torch.no_grad():
                logits = model.predict([(response, ref)])
            pair_scores.append(logits.item())
        scores.append(pair_scores)
    return scores


# ---------------------------------------------------------------------------
# Rule-based evaluation (FaithEval)
# ---------------------------------------------------------------------------

_FAITH_VALID_PHRASES: dict[str, dict[bool, list[str]]] = {
    "unanswerable": {
        False: ["unknown", "no answer", "no information", "not", "unclear"],
        True:  ["unknown"],
    },
    "inconsistent": {
        False: [
            "conflict", "multiple answers", "disagreement", "inconsistent",
            "contradictory", "contradiction", "inconsistency", "two answers",
            "2 answers", "conflicting",
        ],
        True: ["conflict"],
    },
}


def eval_faith(data: list[dict], faith_data: str, strict_match: bool = False) -> float:
    correct = 0
    for sample in data:
        pred = sample["response"].strip()
        if faith_data in ("unanswerable", "inconsistent"):
            phrases = _FAITH_VALID_PHRASES[faith_data][strict_match]
            if any(p in faith_normalize_answer(pred) for p in phrases):
                correct += 1
        elif faith_data == "counterfactual":
            if pred[0].upper() == sample["answer_key"]:
                correct += 1
        else:
            raise ValueError(f"Unknown faith_data: '{faith_data}'")
    return correct / len(data)


# ---------------------------------------------------------------------------
# LLM-based evaluations
# ---------------------------------------------------------------------------

def _build_messages(
    data: list[dict],
    prompt_key: str,
    placeholder_fn,
) -> list[list[dict]]:
    return [
        [{"role": "user", "content": prompt_loader.construct_prompt(prompt_key, placeholders=placeholder_fn(s))}]
        for s in data
    ]


def eval_truthful(
    data: list[dict], args, collect_per_sample: bool = False
) -> tuple[float, int, Optional[list[dict]]]:
    messages = _build_messages(data, "eval_truth", lambda s: {
        "question":        s["question"],
        "correct_answers": "\n".join(s["correct_answers"]),
        "generated_answer": _strip_response(s["response"]),
    })
    eval_results = _llm_batch(messages, args)

    cnt = {"correct": 0, "wrong": 0}
    saved = {}
    per_sample: list[dict] = [] if collect_per_sample else None
    for i, raw in enumerate(eval_results):
        result = _first_alpha_word(raw.strip().lower())
        if args.save_results:
            saved[i] = {"sample": data[i], "eval_result": result, "eval_type": "truth"}
        is_correct = result.startswith("correct")
        if is_correct:
            cnt["correct"] += 1
        elif result.startswith("wrong"):
            cnt["wrong"] += 1
        if collect_per_sample:
            per_sample.append({
                "question_index": data[i].get("question_index", i),
                "is_correct": bool(is_correct),
            })

    if args.save_results:
        _save_results(saved, f"{args.base_model}_{args.decoding_method}_{args.evaluation_type}_{args.data_split}_truth.json")

    total = cnt["correct"] + cnt["wrong"]
    return (cnt["correct"] / total if total else 0.0), total, per_sample


def eval_informativeness(data: list[dict], args) -> tuple[float, int]:
    messages = _build_messages(data, "eval_info", lambda s: {
        "question": s["question"],
        "answer":   s["response"].strip(),
    })
    eval_results = _llm_batch(messages, args)

    cnt = {"complete": 0, "incomplete": 0}
    saved = {}
    for i, raw in enumerate(eval_results):
        result = _first_alpha_word(raw.strip().lower())
        if args.save_results:
            saved[i] = {"sample": data[i], "eval_result": result, "eval_type": "infor"}
        if result.startswith("yes"):
            cnt["complete"] += 1
        elif result.startswith("no"):
            cnt["incomplete"] += 1

    if args.save_results:
        _save_results(saved, f"{args.base_model}_{args.decoding_method}_{args.evaluation_type}_{args.data_split}_infor.json")

    total = cnt["complete"] + cnt["incomplete"]
    return (cnt["complete"] / total if total else 0.0), total


def eval_halu_rate(data: list[dict], args) -> tuple[float, int]:
    _prompt_keys = {
        "halu_qa":  "halu_qa_eval",
        "halu_dia": "halu_dia_eval",
        "halu_sum": "halu_sum_eval",
    }
    if args.eval_data not in _prompt_keys:
        raise ValueError(f"HALU dataset '{args.eval_data}' not supported.")

    messages = _build_messages(data, _prompt_keys[args.eval_data], lambda s: {
        "question": s["question"],
        "answer":   _strip_response(s["response"]),
    })
    eval_results = _llm_batch(messages, args)

    cnt = {"yes": 0, "no": 0}
    saved = {}
    for i, raw in enumerate(eval_results):
        result = _first_alpha_word(raw.strip().lower())
        if args.save_results:
            saved[i] = {"sample": data[i], "eval_result": result, "eval_type": "truth"}
        if result.startswith("yes"):
            cnt["yes"] += 1
        elif result.startswith("no"):
            cnt["no"] += 1

    if args.save_results:
        _save_results(saved, f"{args.base_model}_{args.decoding_method}_{args.evaluation_type}_{args.data_split}_halu.json")

    total = cnt["yes"] + cnt["no"]
    return (cnt["yes"] / total if total else 0.0), total


# ---------------------------------------------------------------------------
# PreciseWiki evaluation
# ---------------------------------------------------------------------------

_ABSTAIN_KEY = "is_abstaining"


def _parse_abstain_response(text: str) -> dict:
    """Try to parse an abstain JSON response robustly."""
    clean = text.replace(" ", "")
    for val in ("false", "true"):
        token = f'{{"{_ABSTAIN_KEY}":{val}}}'
        if token in clean:
            return json.loads(token)
    try:
        return json.loads(text.split("\n")[0])
    except Exception:
        return {"error": "parse_failed"}


def eval_abstain_precisewiki(
    data: list[dict], args
) -> tuple[list[bool], list[str]]:
    messages = _build_messages(data, "precisewiki_abstain", lambda s: {
        "question":         s["question"],
        "generated_answer": s["response"].strip(),
    })
    raw_results = _llm_batch(messages, args)

    refusal = []
    for raw in raw_results:
        parsed = _parse_abstain_response(raw)
        refusal.append(parsed.get(_ABSTAIN_KEY, False))
    return refusal, raw_results


def eval_hallu_precisewiki(data: list[dict], args) -> list[str]:
    messages = _build_messages(data, "precisewiki_hallu", lambda s: {
        "question":         s["question"],
        "generated_answer": s["response"].strip(),
        "correct_answers":  "\n".join(s["correct_answers"]),
    })
    return _llm_batch(messages, args)


def precisewiki_process_res(halu_eval_raw: list[str]) -> list[bool]:
    results = []
    for txt in halu_eval_raw:
        if txt.lower() not in ("correct", "incorrect", "unverifiable"):
            print(f"[PreciseWiki] Unexpected eval output: {txt!r}")
        results.append(txt.lower() not in ("correct", "yes"))
    return results


# ---------------------------------------------------------------------------
# Biography evaluation
# ---------------------------------------------------------------------------

def eval_bio(eval_data_path: str | Path, args) -> dict:
    response_data = json.load(open(eval_data_path))
    with open("data/article.json") as f:
        gt_data = {filter_people(k): v for k, v in json.load(f).items()}

    messages = []
    bio_bullets_list = []

    for sample in response_data:
        person = filter_people(sample["question"])
        if person not in gt_data:
            continue

        gt_bullets = " ".join(parse_bullets(gt_data[person]))
        bio_bullets = parse_bullets(sample["generated_answer"])[:5]
        bio_bullets_list.append(bio_bullets)

        for bullet in bio_bullets:
            messages.append([{"role": "user", "content": (
                f"Reference:\n{gt_bullets}\n\n"
                f"Based on the above reference and your own knowledge about the "
                f"computer scientist {person}, is the following statement correct "
                f"and factual?\n{bullet}\n"
                f"Give a single word answer, yes or no."
            )}])

    eval_results = _llm_batch(messages, args)

    accuracies = [bool(parse_yes_no(r)) if parse_yes_no(r) is not None else False
                  for r in eval_results]

    sample_stats, idx = [], 0
    for bullets in bio_bullets_list:
        n = len(bullets)
        chunk = accuracies[idx : idx + n]
        sample_stats.append({"correct_num": sum(chunk), "incorrect_num": n - sum(chunk)})
        idx += n

    avg_correct   = np.mean([s["correct_num"]   for s in sample_stats])
    avg_incorrect = np.mean([s["incorrect_num"] for s in sample_stats])
    avg_accuracy  = np.mean(accuracies)
    print(f"correct={avg_correct:.2f}  incorrect={avg_incorrect:.2f}  acc={avg_accuracy:.3f}")
    return {"correct_num": avg_correct, "incorrect_num": avg_incorrect, "accuracy": avg_accuracy}


# ---------------------------------------------------------------------------
# Open-generation metrics (ROUGE-L, BERTScore)
# ---------------------------------------------------------------------------

def rouge_l(pred: str, ref: str) -> float:
    return _rouge.score(ref, pred)["rougeL"].fmeasure


def bertscore_f1(
    pred: str,
    ref: str,
    model: str = "microsoft/deberta-xlarge-mnli",
) -> float:
    from bert_score import score as bert_score
    _, _, F1 = bert_score([pred], [ref], model_type=model, lang="en", rescale_with_baseline=True)
    return float(F1[0])


def eval_open(data: list[dict]) -> dict[str, float]:
    rouge_scores, bert_scores = [], []
    for sample in data:
        gt = sample["correct_answers"]
        gt = gt[0] if isinstance(gt, list) and gt else gt
        if not gt:
            continue
        rouge_scores.append(rouge_l(sample["response"], gt))
        bert_scores.append(bertscore_f1(sample["response"], gt))

    def _mean(lst): return sum(lst) / len(lst) if lst else 0.0
    return {"rougeL": _mean(rouge_scores), "bertscore_f1": _mean(bert_scores)}


# ---------------------------------------------------------------------------
# Unified eval_metric dispatcher
# ---------------------------------------------------------------------------

def eval_metric(data: list[dict], args, faith_data: Optional[str] = None) -> dict:
    m = args.eval_metric

    collect_ps = getattr(args, "save_per_sample", False)

    if m == "factuality":
        truth_score, _, truth_ps = eval_truthful(data, args, collect_per_sample=collect_ps)
        info_score, _  = eval_informativeness(data, args)
        out = {
            "t_times_i":   truth_score * info_score,
            "truth_score": truth_score,
            "info_score":  info_score,
            "n_samples":   len(data),
        }
        if truth_ps is not None:
            out["_per_sample"] = truth_ps
        return out

    if m == "truth_only":
        truth_score, valid, truth_ps = eval_truthful(data, args, collect_per_sample=collect_ps)
        out = {"truth_score": truth_score, "truth_valid": valid, "n_samples": len(data)}
        if truth_ps is not None:
            out["_per_sample"] = truth_ps
        return out

    if m == "halu_rate":
        rate, valid = eval_halu_rate(data, args)
        return {"halu_rate": rate, "halu_valid": valid, "n_samples": len(data)}

    if m == "faith":
        if not args.eval_data.startswith("faith"):
            raise ValueError("faith metric requires a FaithEval dataset.")
        fd = faith_data or args.eval_data.split("_", 1)[1]
        return {
            "faith_score":        eval_faith(data, fd, strict_match=False),
            "faith_score_strict": eval_faith(data, fd, strict_match=True),
            "n_samples":          len(data),
        }

    if m == "open":
        scores = eval_open(data)
        return {**scores, "n_samples": len(data)}

    if m == "precisewiki":
        if args.eval_data != "precisewiki":
            raise ValueError("precisewiki metric requires the precisewiki dataset.")
        abstain_res, _ = eval_abstain_precisewiki(data, args)
        halu_raw       = eval_hallu_precisewiki(data, args)
        halu_res       = precisewiki_process_res(halu_raw)

        not_abstained = sum(1 for x in abstain_res if not x)
        hallu_given_answered = (
            sum(1 for a, h in zip(abstain_res, halu_res) if not a and h) / not_abstained
            if not_abstained else 0.0
        )
        result = {
            "halu_rate":    hallu_given_answered,
            "refusal_rate": sum(abstain_res) / len(abstain_res),
            "correct_rate": sum(1 for h in halu_res if not h) / len(halu_res),
        }
        if getattr(args, "save_per_sample", False):
            result["_per_sample"] = [
                {
                    "question_index": data[i].get("question_index", i),
                    "is_hallucinated": halu_res[i],
                    "is_abstaining":   abstain_res[i],
                }
                for i in range(len(data))
            ]
        return result

    if m == "hhem":
        return {"hhem_score": eval_hhem(data), "n_samples": len(data)}

    raise ValueError(f"Unknown eval_metric: '{m}'")


# ---------------------------------------------------------------------------
# Dataset loader for evaluation
# ---------------------------------------------------------------------------

_EVAL_LOADERS = {
    "truthful_qa": load_truthfulqa,
    "precisewiki":  load_precisewiki,
    "wiki":         load_wiki,
    "alpaca":       load_alpaca,
    "bio":          load_biography,
    "halu_qa":      load_halu_qa,
    "halu_dia":     load_halu_dia,
    "halu_sum":     load_halu_sum,
}


def load_data_eval(args) -> list[dict]:
    if args.eval_data.startswith("faith"):
        faith_key = args.eval_data.split("_", 1)[1]
        return load_faith(
            split=args.data_split, faith_data=faith_key,
            max_sample_num=args.max_sample_num,
        )
    if args.eval_data not in _EVAL_LOADERS:
        raise ValueError(f"Evaluation data '{args.eval_data}' not supported.")
    return _EVAL_LOADERS[args.eval_data](
        split=args.data_split, max_sample_num=args.max_sample_num
    )


# ---------------------------------------------------------------------------
# Single-file evaluation entry point
# ---------------------------------------------------------------------------

def eval_single_file(file_path: str | Path, args) -> dict:
    response_data = json.load(open(file_path))

    # Biography uses its own path-based evaluator
    if args.eval_data == "bio":
        metric_dict = eval_bio(eval_data_path=file_path, args=args)
    else:
        data = load_data_eval(args)[: len(response_data)]
        for i, rd in enumerate(response_data):
            data[i]["response"] = rd["generated_answer"]
            data[i]["question_index"] = rd.get("question_index", i)
        metric_dict = eval_metric(data=data, args=args)

    # Save per-sample results to a separate file if requested
    per_sample = metric_dict.pop("_per_sample", None)
    if per_sample is not None and getattr(args, "per_sample_out", None):
        Path(args.per_sample_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.per_sample_out, "w") as _f:
            json.dump(per_sample, _f, indent=2)
        print(f"[PerSample saved] {args.per_sample_out}")

    print(f"Eval results for {file_path}: {metric_dict}")

    result = {
        "train_data":       args.train_data,
        "eval_data":        args.eval_data,
        "data_split":       args.data_split,
        "sample_num":       args.max_sample_num,
        "model":            args.base_model,
        "max_new_tokens":   args.max_new_tokens,
        "decoding_method":  args.decoding_method,
        "eval_metric":      args.eval_metric,
        "evaluation_type":  args.evaluation_type,
        "exp_tag":          getattr(args, "exp_tag", ""),
        "eval_data_path":   args.eval_data_path,
        "eval_results":     {f"metric__{k}": v for k, v in metric_dict.items()},
        **{
            f"{args.decoding_method}__{k}": v
            for k, v in args.decoding_config.items()
        },
        "timestamp": time(),
        "run_id":    f"{args.base_model}__{args.decoding_method}__{args.seed}",
    }

    result_path = Path(result_dir) / f"{args.eval_data}_{args.base_model}_{args.decoding_method}.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a") as f:
        f.write(json.dumps(result) + "\n")

    return result