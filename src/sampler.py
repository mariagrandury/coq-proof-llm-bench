"""
Proof sampling module for Coq proof generation using either a baseline proof or a Hugging Face model.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .prompt import build_prompt
from .schemas import GenConfig, LemmaSpec

_BLOCK_PATTERN = re.compile(r"Proof\.(.*?)Qed\.", re.S)


@dataclass
class HFSession:
    """Hugging Face model session containing tokenizer and model.

    Attributes:
        tokenizer: The tokenizer for text processing
        model: The language model for text generation
    """

    tokenizer: Any
    model: Any


@lru_cache(maxsize=10)
def _load_hf_session(model_id: str) -> HFSession:
    """Load and cache Hugging Face model session.

    Uses LRU cache to avoid reloading models for the same model_id.
    Maximum of 10 different models can be cached simultaneously.

    Args:
        model_id: Hugging Face model identifier

    Returns:
        HFSession containing the loaded tokenizer and model

    Raises:
        ImportError: If transformers library is not available
        RuntimeError: If model loading fails
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("transformers library is required for HF backend")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        return HFSession(tokenizer=tokenizer, model=model)
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_id}: {e}")


def _generate_hf_proof(
    session: HFSession, prompt: str, cfg: GenConfig, try_idx: int
) -> str:
    """Generate proof using Hugging Face model with given configuration.

    Args:
        session: Loaded HF session with tokenizer and model
        prompt: Input prompt for proof generation
        cfg: Generation configuration parameters
        try_idx: Index for reproducible random seed

    Returns:
        Generated proof text

    Raises:
        RuntimeError: If generation fails
    """
    import torch

    # Set per-try seed for reproducible generation
    torch.manual_seed(try_idx)

    # Tokenize input
    input_ids = session.tokenizer(prompt, return_tensors="pt").to(session.model.device)

    # Prepare end-of-sequence token
    eos_tokens = session.tokenizer.encode("Qed.", add_special_tokens=False)
    eos_token_id = eos_tokens[0] if eos_tokens else None

    try:
        # Generate text
        output = session.model.generate(
            **input_ids,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            eos_token_id=eos_token_id,
        )

        # Decode generated text (excluding input prompt)
        input_length = input_ids["input_ids"].shape[1]
        generated_text = session.tokenizer.decode(
            output[0][input_length:], skip_special_tokens=True
        )

        return generated_text

    except Exception as e:
        raise RuntimeError(f"Text generation failed: {e}")


def _normalize_to_block(proof_text: str) -> str:
    """Normalize proof text to ensure it has proper Proof/Qed structure.

    Args:
        proof_text: Raw proof text that may or may not have Proof/Qed markers

    Returns:
        Normalized proof text with proper Proof/Qed structure

    Example:
        >>> _normalize_to_block("auto.")
        "Proof.\\nauto.\\nQed."
    """
    # Try to extract existing Proof/Qed block
    match = _BLOCK_PATTERN.search(proof_text)
    if match:
        return f"Proof.{match.group(1)}Qed."

    # Clean and normalize the text
    body = proof_text.strip()

    # Ensure proper ending
    if not body.endswith("Qed."):
        body += "\nQed."

    # Ensure proper beginning
    if not body.startswith("Proof."):
        body = "Proof.\n" + body

    return body


def sample_proof(lemma: LemmaSpec, cfg: GenConfig, try_idx: int, model_id: str) -> str:
    """Generate a proof for a given lemma using the specified backend.

    This is a stateless sampler that can use either a baseline proof or
    generate new proofs using Hugging Face language models. For HF backend,
    models are automatically cached to avoid reloading.

    Args:
        lemma: Lemma specification containing problem details
        cfg: Generation configuration (backend, parameters, etc.)
        try_idx: Index for reproducible random seed
        model_id: Model identifier (used for HF backend)

    Returns:
        Generated proof text normalized to Proof/Qed format

    Example:
        >>> config = GenConfig(backend="hf", max_new_tokens=256)
        >>> proof = sample_proof(lemma, config, 0, "gpt2")
        >>> print(proof)
        "Proof.\\nauto.\\nQed."
    """
    # Handle baseline backend
    if cfg.backend == "baseline":
        baseline_proof = getattr(lemma, "baseline_proof", "Proof. Fail trivial. Qed.")
        return "Prompt", _normalize_to_block(baseline_proof)

    # Handle Hugging Face backend
    if cfg.backend == "hf":
        session = _load_hf_session(model_id)
        prompt = build_prompt(lemma)
        generated_text = _generate_hf_proof(session, prompt, cfg, try_idx)
        return prompt, _normalize_to_block(generated_text)
