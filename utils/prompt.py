# utils/prompt.py
"""
PromptTemplateLoader - Single YAML file version
All prompts are centralized in all_prompts.yaml
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Optional, Dict, Any

import yaml

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Strictly follow the given instruction to generate a response."
)

_THINKING_PARAM = "enable_thinking"


class PromptTemplateLoader:
    """
    Loads all prompts from a single YAML file (all_prompts.yaml).
    Supports prompt groups and multiple templates per group.
    """

    def __init__(self, prompts_file: str | Path = "DEFAULT"):
        """
        Args:
            prompts_file: Path to all_prompts.yaml
        """
        if prompts_file == "DEFAULT":
            prompts_file = Path(__file__).parent / "all_prompts.yaml"

        self.prompts_file = Path(prompts_file)
        self._cache: Dict[str, Any] = None
        self._load_all_prompts()

    def _load_all_prompts(self):
        """Load the entire all_prompts.yaml file."""
        if not self.prompts_file.exists():
            raise FileNotFoundError(f"Prompts file not found: {self.prompts_file}")

        with self.prompts_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self._cache = data.get("prompts", {})
        if not self._cache:
            raise ValueError("Invalid prompts.yaml: 'prompts' section is missing or empty.")

    def construct_prompt(
        self,
        prompt_group: str,
        template_name: str = "zero_shot",
        placeholders: dict | None = None
    ) -> str:
        """
        Build prompt text from prompt_group and template_name.

        Example: construct_prompt("truthful_qa", "cot", {"question": "..."})
        """
        placeholders = placeholders or {}

        if prompt_group not in self._cache:
            raise KeyError(f"Prompt group '{prompt_group}' not found in all_prompts.yaml")

        group = self._cache[prompt_group]

        # Case 1: Group has direct template (simple groups)
        if "template" in group:
            template_str = group["template"]
        # Case 2: Group has multiple templates
        elif "templates" in group:
            templates = group["templates"]
            resolved = template_name if template_name in templates else (
                "default" if "default" in templates else None
            )
            if resolved is None:
                raise KeyError(f"Template '{template_name}' not found in group '{prompt_group}'. "
                               f"Available: {list(templates.keys())}")
            template_str = templates[resolved].get("template")
        else:
            raise ValueError(f"Invalid structure for prompt group '{prompt_group}'")

        if not template_str:
            raise ValueError(f"Template string is empty for '{prompt_group}.{template_name}'")

        return template_str.format(**placeholders)

    def get_system_instruction(
        self,
        prompt_group: str,
        template_name: str = "zero_shot",
        placeholders: dict | None = None
    ) -> str:
        """Retrieve system instruction for the given prompt group."""
        placeholders = placeholders or {}
        group = self._cache.get(prompt_group, {})

        if "system_instruction" in group:
            sys_instr = group["system_instruction"]
        elif "templates" in group:
            templates = group["templates"]
            resolved = template_name if template_name in templates else (
                "default" if "default" in templates else None
            )
            sys_instr = templates[resolved].get("system_instruction") if resolved else None
        else:
            sys_instr = None

        return sys_instr.format(**placeholders) if sys_instr else DEFAULT_SYSTEM_PROMPT

    def construct_chat_input(
        self,
        prompt_group: str,
        template_name: str = "zero_shot",
        placeholders: dict | None = None,
        tokenizer=None,
        system_prompt: Optional[str] = None,
    ) -> str | list[dict]:
        """Build chat-formatted input ready for tokenizer. Returns raw chat list if tokenizer is None."""
        placeholders = placeholders or {}

        user_message = self.construct_prompt(prompt_group, template_name, placeholders)
        system = system_prompt or self.get_system_instruction(prompt_group, template_name, placeholders)

        chat = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_message},
        ]

        if tokenizer is None:
            return chat

        # Some models do not support system role
        if "System role not supported" in getattr(tokenizer, "chat_template", ""):
            chat = [{"role": "user", "content": user_message}]

        apply_kwargs = {"tokenize": False, "add_generation_prompt": True}

        if _THINKING_PARAM in inspect.signature(tokenizer.apply_chat_template).parameters:
            apply_kwargs[_THINKING_PARAM] = False

        return tokenizer.apply_chat_template(chat, **apply_kwargs)

    def list_prompt_groups(self) -> list[str]:
        """Return all available prompt groups."""
        return list(self._cache.keys())