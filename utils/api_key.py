"""
utils/api_key.py — API credentials.

Priority (highest → lowest):
  1. Environment variable  COHERE_API_KEY / GEMINI_API_KEY
  2. File  local/keys.env  (key=value lines, one per line)

Create local/keys.env from the template:
    cp local/keys.env.template local/keys.env
    # then fill in your keys
"""
import os
from pathlib import Path


def _load_keys_file() -> dict:
    path = Path(__file__).resolve().parent.parent / "local" / "keys.env"
    if not path.exists():
        return {}
    keys = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            keys[k.strip()] = v.strip()
    return keys


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key) or _load_keys_file().get(key, default)


cohere_api_key    = _get("COHERE_API_KEY")
cohere_model      = _get("COHERE_MODEL",      "command-a-03-2025")
cohere_model_test = _get("COHERE_MODEL_TEST", "command-r-08-2024")

gemini_api_key = _get("GEMINI_API_KEY")
gemini_model   = _get("GEMINI_MODEL", "gemini-2.0-flash")
