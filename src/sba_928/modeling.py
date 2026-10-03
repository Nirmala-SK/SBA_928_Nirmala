"""Model loading and deterministic generation shared by steps 2, 4, and 6."""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> str:
    """Return 'cuda', 'mps', or 'cpu'. Raises if a specific device is unavailable."""
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unknown device '{requested}'")
    return requested


def generate_from_model_path(
    model_path: str | Path,
    prompts: list[str],
    requested_device: str = "auto",
    max_new_tokens: int = 96,
    max_source_length: int = 256,
    batch_size: int = 8,
) -> list[str]:
    """Greedy (deterministic) generation. Decoding settings are fixed on purpose
    so base and fine-tuned models can be compared fairly."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = resolve_device(requested_device)
    LOGGER.info("Loading %s on %s", model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path)).to(device)
    model.eval()

    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_length,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return [text.strip() for text in outputs]