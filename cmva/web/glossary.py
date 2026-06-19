"""Glossary loading for web tooltips."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


GLOSSARY_PATH = Path(__file__).with_name("glossary.yaml")


@lru_cache(maxsize=1)
def load_glossary() -> dict[str, dict[str, str]]:
    raw = yaml.safe_load(GLOSSARY_PATH.read_text(encoding="utf-8")) or {}
    return {str(key): {str(item_key): str(item_value) for item_key, item_value in value.items()} for key, value in raw.items()}
