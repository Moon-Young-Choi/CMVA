"""Volatility model registry."""

from __future__ import annotations

from cmva.models.garch import GarchVolatilityModel

MODEL_REGISTRY = {"garch": GarchVolatilityModel}


def create_model(name: str = "garch", **kwargs):
    try:
        model_type = MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown model: {name}") from exc
    return model_type(**kwargs)
