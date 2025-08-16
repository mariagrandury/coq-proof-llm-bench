"""
Generate proofs for Coq lemmas using language models or baseline proofs.

This script generates multiple proof candidates for each lemma using either:
- Hugging Face language models (backend='hf')
- Pre-defined baseline proofs (backend='baseline')

The script supports batch processing of multiple models and outputs results
in JSONL format for further analysis.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from .sampler import sample_proof
from .schemas import GenConfig, LemmaSpec
from .utils import sanitize_dirname, write_jsonl


def load_lemmas(path: str) -> List[LemmaSpec]:
    """Load lemmas from JSONL file.

    Args:
        path: Path to JSONL file containing lemma specifications

    Returns:
        List of LemmaSpec objects
    """
    items: List[LemmaSpec] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                items.append(LemmaSpec(**obj))
    return items


def create_proof_row(
    lemma: LemmaSpec, proof: str, model_id: str, cfg: GenConfig, try_idx: int
) -> Dict[str, Any]:
    """Create a proof result row for output.

    Args:
        lemma: Lemma specification
        proof: Generated proof text
        model_id: Model identifier used
        cfg: Generation configuration
        try_idx: Attempt index

    Returns:
        Dictionary containing all proof metadata and results
    """
    return {
        "id": lemma.id,
        "lang": getattr(lemma, "lang", None),
        "category": getattr(lemma, "category", None),
        "difficulty": getattr(lemma, "difficulty", None),
        "phenomena": getattr(lemma, "phenomena", []),
        "model": model_id if cfg.backend == "hf" else "baseline",
        "try_idx": try_idx,
        "gen_params": {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_new_tokens": cfg.max_new_tokens,
        },
        "coq_prelude": getattr(lemma, "coq_prelude", []),
        "statement": lemma.statement,
        "proof": proof,
    }


def generate_for_model(
    lemmas: List[LemmaSpec],
    model_id: str,
    cfg: GenConfig,
    outdir: str,
) -> str:
    """Generate proof candidates for a single model.

    Generates k proof candidates per lemma and writes results to proofs.jsonl.

    Args:
        lemmas: List of lemmas to process
        model_id: Model identifier (empty string for baseline)
        cfg: Generation configuration
        outdir: Base output directory

    Returns:
        Path to the generated proofs file

    Raises:
        OSError: If unable to create directories or write files
    """
    rows: List[Dict[str, Any]] = []

    # Generate proofs for each lemma
    for lemma in lemmas:
        for try_idx in range(cfg.k):
            proof = sample_proof(lemma, cfg, try_idx=try_idx, model_id=model_id)
            rows.append(create_proof_row(lemma, proof, model_id, cfg, try_idx))

    # Create output directory and write results
    model_name = model_id if model_id else "baseline"
    model_dir = os.path.join(outdir, sanitize_dirname(model_name))
    os.makedirs(model_dir, exist_ok=True)

    proofs_path = os.path.join(model_dir, "proofs.jsonl")
    write_jsonl(proofs_path, rows)
    return proofs_path


def main():
    """Main entry point for proof generation."""
    parser = argparse.ArgumentParser(
        description="Generate proof candidates using language models or baseline proofs."
    )

    # Required arguments
    parser.add_argument("--lemmas", required=True, help="Path to lemmas JSONL file")

    # Backend selection
    parser.add_argument(
        "--backend",
        choices=["baseline", "hf"],
        default="hf",
        help="Backend: 'hf' for HuggingFace models, 'baseline' for built-in proofs",
    )

    # Model selection (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", help="Single HuggingFace model ID")
    model_group.add_argument(
        "--models_file", help="Text file with one model ID per line"
    )

    # Output configuration
    parser.add_argument(
        "--outdir",
        default="results/batch",
        help="Output directory (default: results/batch)",
    )

    # Generation parameters
    parser.add_argument(
        "-k", type=int, default=5, help="Number of proof samples per lemma (default: 5)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)",
    )

    # Processing options
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of lemmas to process (0=all, default: 0)",
    )

    args = parser.parse_args()

    # Load and optionally limit lemmas
    print(f"[GEN] Loading lemmas from {args.lemmas}")
    lemmas = load_lemmas(args.lemmas)
    if args.limit:
        lemmas = lemmas[: args.limit]
        print(f"[GEN] Limited to {len(lemmas)} lemmas")
    else:
        print(f"[GEN] Processing {len(lemmas)} lemmas")

    # Determine models to process
    if args.models_file:
        with open(args.models_file, "r", encoding="utf-8") as f:
            models = [line.strip() for line in f if line.strip()]
        print(f"[GEN] Loaded {len(models)} models from {args.models_file}")
    else:
        models = [args.model] if args.model else []
        print(f"[GEN] Using single model: {args.model}")

    # Create shared configuration
    cfg = GenConfig(
        backend=args.backend,
        model_name=None,  # Not used in this script
        k=args.k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(
        f"[GEN] Configuration: backend={cfg.backend}, k={cfg.k}, "
        f"temp={cfg.temperature}, top_p={cfg.top_p}"
    )

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Process each model
    models_to_process = models if args.backend == "hf" else ["baseline"]
    for model_id in models_to_process:
        print(f"[GEN] Generating for: {model_id}")

        # For baseline backend, pass empty string as model_id
        actual_model_id = model_id if args.backend == "hf" else ""

        try:
            proofs_path = generate_for_model(
                lemmas=lemmas,
                model_id=actual_model_id,
                cfg=cfg,
                outdir=args.outdir,
            )
            print(f"[GEN] Wrote proofs → {proofs_path}")
        except Exception as e:
            print(f"[ERROR] Failed to generate proofs for {model_id}: {e}")
            continue


if __name__ == "__main__":
    main()
