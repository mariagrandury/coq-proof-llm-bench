from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Any, Dict, List

from .assemble import assemble_coq_file
from .check_coq import check_with_coqc
from .utils import read_jsonl, write_jsonl


def verify_proofs_file(proofs_path: str, timeout: int) -> Dict[str, Any]:
    """
    Verify one proofs.jsonl and write results+summary next to it.
    Returns summary dict.
    """
    print(f"[VER] Reading {proofs_path}")
    cand = read_jsonl(proofs_path)

    # Group tries by lemma; early stop on first success
    by_lemma: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in cand:
        by_lemma[r["id"]].append(r)

    # Flattened results
    all_rows: List[Dict[str, Any]] = []

    for lid, tries in by_lemma.items():
        tries = sorted(tries, key=lambda x: x["try_idx"])
        for t in tries:
            # Assemble a .v in-memory and check with Coq
            coq_text = assemble_coq_file(
                type(
                    "L",
                    (),
                    {
                        "coq_prelude": t["coq_prelude"],
                        "statement": t["statement"],
                        "requires_classical": False,
                    },
                )(),
                t["proof"],
            )
            ok, error = check_with_coqc(coq_text, timeout_sec=timeout)
            all_rows.append(
                {
                    "id": lid,
                    "lang": t.get("lang"),
                    "category": t.get("category"),
                    "difficulty": t.get("difficulty"),
                    "ok": ok,
                    "stderr": error,
                    "model": t.get("model"),
                    "try_idx": t["try_idx"],
                    "proof": t["proof"],
                }
            )
            if ok:
                break

    # Write results.jsonl in same folder
    mdir = os.path.dirname(proofs_path)
    results_path = os.path.join(mdir, "results.jsonl")
    write_jsonl(results_path, all_rows)
    print(f"[VER] Wrote {results_path}")
    return {"n_lemmas": len(by_lemma), "n_results": len(all_rows)}


def main():
    ap = argparse.ArgumentParser(
        description="Verify proofs with Coq (local), write results+summary."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--proofs", help="Path to a single proofs.jsonl file.")
    g.add_argument(
        "--proofs_dir",
        help="Directory containing per-model subfolders with proofs.jsonl.",
    )
    ap.add_argument(
        "--timeout", type=int, default=15, help="Per-proof coqc timeout (seconds)."
    )
    args = ap.parse_args()

    if args.proofs:
        # Single file mode
        verify_proofs_file(args.proofs, timeout=args.timeout)
        return

    # Directory mode
    models_dirs = [
        os.path.join(args.proofs_dir, d) for d in os.listdir(args.proofs_dir)
    ]
    for mdir in models_dirs:
        proofs_path = os.path.join(mdir, "proofs.jsonl")
        if not os.path.exists(proofs_path):
            print(f"[SKIP] No proofs.jsonl in {mdir}")
            continue
        verify_proofs_file(proofs_path, timeout=args.timeout)


if __name__ == "__main__":
    main()
