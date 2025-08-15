import re
from dataclasses import dataclass
from typing import Any

from .prompt import build_prompt
from .schemas import GenConfig, LemmaSpec

_BLOCK_RE = re.compile(r"Proof\.(.*?)Qed\.", re.S)


def _normalize_to_block(s: str) -> str:
    m = _BLOCK_RE.search(s)
    if m:
        return f"Proof.{m.group(1)}Qed."
    body = s.strip()
    if not body.endswith("Qed."):
        body += "\nQed."
    if not body.startswith("Proof."):
        body = "Proof.\n" + body
    return body


@dataclass
class HFSession:
    tok: Any
    model: Any


def _load_hf_session(model_id: str) -> HFSession:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    return HFSession(tok=tok, model=model)


def sample_proof(lemma: LemmaSpec, cfg: GenConfig, try_idx: int, model_id: str) -> str:
    """
    Stateless sampler. For backend='hf' pass a loaded HFSession to avoid reloading per sample.
    """
    if cfg.backend == "baseline":
        bp = getattr(lemma, "baseline_proof", None)
        return _normalize_to_block(bp or "Proof. Fail trivial. Qed.")
    if cfg.backend == "hf":
        session = _load_hf_session(model_id)
    else:
        raise ValueError("HF session required for backend='hf'")
    import torch

    torch.manual_seed(try_idx)  # per-try seed
    prompt = build_prompt(lemma)
    input_ids = session.tok(prompt, return_tensors="pt").to(session.model.device)
    eos = session.tok.encode("Qed.", add_special_tokens=False)
    eos_id = eos[0] if eos else None
    out = session.model.generate(
        **input_ids,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        eos_token_id=eos_id,
    )
    gen = session.tok.decode(
        out[0][input_ids["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return _normalize_to_block(gen)
