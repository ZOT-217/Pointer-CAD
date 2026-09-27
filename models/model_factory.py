"""Explicit model-mode factory shared by smoke tooling and future training."""
from __future__ import annotations

from .crs_pointercad import ModelMode, build_pointercad


def from_config(config: dict):
    mode = config.get("mode", ModelMode.LEGACY_POINTERCAD.value)
    model_config = dict(config.get("model", {}))
    if mode == ModelMode.LEGACY_POINTERCAD.value:
        # Legacy config calls this field base_model; keep the old constructor API.
        if "base_model" in model_config:
            model_config["qwen_model"] = model_config.pop("base_model")
    else:
        # CRS-expanded mode owns the same Qwen loading boundary as legacy
        # PointerCAD.  Keep the configured base_model live.
        if "base_model" in model_config:
            model_config["qwen_model"] = model_config.pop("base_model")
    return build_pointercad(mode=mode, **model_config)
