from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from .sampler import sample_proof
from .schemas import GenConfig, LemmaSpec
from .utils import sanitize_dirname, write_jsonl


def load_lemmas(path: str) -> List[LemmaSpec]:
    items: List[LemmaSpec] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                items.append(LemmaSpec(**obj))
    return items


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

    for lemma in lemmas:
        for i in range(cfg.k):
            proof = sample_proof(lemma, cfg, try_idx=i, model_id=model_id)
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

    for model_id in models if args.backend == "hf" else ["baseline"]:
        print(f"[GEN] Generating for: {model_id}")
        # For HF we pass the model id via generate_for_model; for baseline model_id is 'baseline'
        proofs_path = generate_for_model(
            lemmas=lemmas,
            model_id=model_id if args.backend == "hf" else "",
            cfg=cfg,
            outdir=args.outdir,
        )
        print(f"[GEN] Wrote proofs → {proofs_path}")


if __name__ == "__main__":
    main()
