import re
from typing import Optional

from .schemas import GenConfig, LemmaSpec

# ——— helpers ———
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


# ——— backends ———


def gen_proof(lemma: LemmaSpec, cfg: GenConfig, seed: int) -> str:
    if cfg.backend == "baseline":
        if lemma.baseline_proof:
            return _normalize_to_block(lemma.baseline_proof)
        return "Proof. Fail trivial. Qed."  # will fail; for testing
    elif cfg.backend == "hf":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            raise RuntimeError("Install transformers to use hf backend") from e
        tok = AutoTokenizer.from_pretrained(cfg.model_name)
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, device_map="auto")
        torch.manual_seed(seed)
        prompt = build_prompt(lemma)  # local import to avoid circular
        # import here to avoid circular dependencies
        input_ids = tok(prompt, return_tensors="pt").to(model.device)
        eos_id = tok.encode("Qed.", add_special_tokens=False)
        eos_id = eos_id[0] if eos_id else None
        out = model.generate(
            **input_ids,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            eos_token_id=eos_id,
        )
        gen = tok.decode(
            out[0][input_ids["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return _normalize_to_block(gen)
    else:
        raise ValueError(f"Unknown backend {cfg.backend}")


# delayed import to avoid cycle
from .prompt import build_prompt
