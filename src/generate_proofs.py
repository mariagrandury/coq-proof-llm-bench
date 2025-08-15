from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .prompt import build_prompt
from .schemas import GenConfig, LemmaSpec

# ------------------------- IO helpers -------------------------


def load_lemmas(path: str) -> List[LemmaSpec]:
    items: List[LemmaSpec] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                items.append(LemmaSpec(**obj))
    return items


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sanitize_dirname(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


# ------------------------- Sampler ---------------------------

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


def sample_proof(
    lemma: LemmaSpec, cfg: GenConfig, try_idx: int, session: Optional[HFSession]
) -> str:
    """
    Stateless sampler. For backend='hf' pass a loaded HFSession to avoid reloading per sample.
    """
    if cfg.backend == "baseline":
        bp = getattr(lemma, "baseline_proof", None)
        return _normalize_to_block(bp or "Proof. Fail trivial. Qed.")
    # hf
    assert session is not None, "HF session required for backend='hf'"
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


# ------------------------- Main logic ------------------------


def generate_for_model(
    lemmas: List[LemmaSpec],
    model_id: str,
    cfg: GenConfig,
    outdir: str,
) -> str:
    """
    Generate k proof candidates per lemma for a single model and write to proofs.jsonl.
    Returns the path written.
    """
    rows: List[Dict[str, Any]] = []

    if cfg.backend == "hf":
        session = _load_hf_session(model_id)
    else:
        session = None

    for lemma in lemmas:
        for i in range(cfg.k):
            proof = sample_proof(lemma, cfg, try_idx=i, session=session)
            rows.append(
                {
                    "id": lemma.id,
                    "lang": getattr(lemma, "lang", None),
                    "category": getattr(lemma, "category", None),
                    "difficulty": getattr(lemma, "difficulty", None),
                    "phenomena": getattr(lemma, "phenomena", []),
                    "model": model_id if cfg.backend == "hf" else "baseline",
                    "try_idx": i,
                    "gen_params": {
                        "temperature": cfg.temperature,
                        "top_p": cfg.top_p,
                        "max_new_tokens": cfg.max_new_tokens,
                    },
                    "coq_prelude": getattr(lemma, "coq_prelude", []),
                    "statement": lemma.statement,
                    "proof": proof,
                }
            )

    model_dir = os.path.join(
        outdir, sanitize_dirname(model_id if model_id else "baseline")
    )
    os.makedirs(model_dir, exist_ok=True)
    proofs_path = os.path.join(model_dir, "proofs.jsonl")
    write_jsonl(proofs_path, rows)
    return proofs_path


def main():
    ap = argparse.ArgumentParser(
        description="Generate proof candidates (GPU-friendly, no Coq)."
    )
    ap.add_argument("--lemmas", required=True, help="Path to lemmas JSONL.")
    ap.add_argument(
        "--backend",
        choices=["baseline", "hf"],
        default="hf",
        help="Use 'hf' for HuggingFace models or 'baseline' for built-in proofs.",
    )
    g = ap.add_mutually_exclusive_group(required=(True))
    g.add_argument("--model", help="HF model id (when generating for a single model).")
    g.add_argument("--models_file", help="Text file with one HF model id per line.")
    ap.add_argument(
        "--outdir",
        default="results/batch",
        help="Output directory (per-model subfolders).",
    )
    ap.add_argument("-k", type=int, default=5, help="Proof samples per lemma.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument(
        "--limit", type=int, default=0, help="Limit number of lemmas (0=all)."
    )
    args = ap.parse_args()

    # Load lemmas
    lemmas = load_lemmas(args.lemmas)
    if args.limit:
        lemmas = lemmas[: args.limit]

    # Build list of models
    if args.models_file:
        with open(args.models_file, "r", encoding="utf-8") as f:
            models = [ln.strip() for ln in f if ln.strip()]
    else:
        models = [args.model] if args.model else []

    # Config (shared across models)
    cfg = GenConfig(
        backend=args.backend,
        model_name=(
            None if args.backend == "baseline" else None
        ),  # not needed here; we pass model dynamically
        k=args.k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    os.makedirs(args.outdir, exist_ok=True)

    for mid in models if args.backend == "hf" else ["baseline"]:
        print(f"[GEN] Generating for: {mid}")
        # For HF we pass the model id via generate_for_model; for baseline mid is 'baseline'
        proofs_path = generate_for_model(
            lemmas=lemmas,
            model_id=mid if args.backend == "hf" else "",
            cfg=cfg,
            outdir=args.outdir,
        )
        print(f"[GEN] Wrote proofs → {proofs_path}")


if __name__ == "__main__":
    main()
