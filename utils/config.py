"""
utils/config.py — project-wide paths and model registry loaded from config.yaml.

Path resolution priority (highest → lowest):
  1. Environment variable  DECODEHUB_<KEY>  (e.g. DECODEHUB_OUTPUT=/mnt/fast/out)
  2. Entry in config.yaml  [paths] section
  3. Built-in default (relative to the project root where config.yaml lives)

API keys must be set as environment variables; see utils/api_key.py.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH  = _PROJECT_ROOT / "config.yaml"


def _load() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    with _CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_cfg: dict       = _load()
_paths_cfg: dict = _cfg.get("paths", {})


def _resolve_path(cfg_key: str, default: str) -> str:
    """Return an absolute path string, respecting env → config → default."""
    env_key = f"DECODEHUB_{cfg_key.upper()}"
    raw = os.environ.get(env_key) or _paths_cfg.get(cfg_key) or default
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (_PROJECT_ROOT / p).resolve()
    return str(p)


# ---------------------------------------------------------------------------
# Exported path constants (names kept for backwards compatibility)
# ---------------------------------------------------------------------------

hf_cache_dir      = _resolve_path("hf_cache",    "~/.cache/huggingface")
hf_datasets_cache = hf_cache_dir
output_dir        = _resolve_path("output",       "output")
data_dir          = _resolve_path("data",         "data")
result_dir        = _resolve_path("results",      "results")
eval_result_dir   = _resolve_path("eval_results", "evaluation_results")

# Raw PreciseWiki JSONL path (empty string means "not configured").
precisewiki_raw: str = _paths_cfg.get("precisewiki_raw") or ""

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_PATH_DICT: dict[str, str] = _cfg.get("models", {})


def get_model_path(alias: str) -> str:
    """Resolve a short model alias to a HuggingFace repo id or local path.

    If *alias* is already a full repo id (contains '/') or an existing local
    path it is returned unchanged, so scripts that hard-code a full path still
    work without changes.
    """
    if alias in MODEL_PATH_DICT:
        return MODEL_PATH_DICT[alias]
    if "/" in alias or Path(alias).exists():
        return alias
    raise ValueError(
        f"Model alias '{alias}' is not listed in config.yaml [models] and is "
        "not a recognised HuggingFace repo id or local path."
    )
