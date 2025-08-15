from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

from .assemble import assemble_coq_file
from .check_coq import check_with_coqc
from .utils import read_jsonl, write_jsonl


def summarize_passk(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    rows: flattened per-try verification results including fields:
      id, ok, lang, category, difficulty, phenomena?, model, try_idx
    Computes pass@1 and pass@k by lemma, then aggregates simple stratified stats.
    """
    by_lemma: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_lemma[r["id"]].append(r)

    first_success: Dict[str, Dict[str, bool]] = {}
    for lid, tries in by_lemma.items():
        tries_sorted = sorted(tries, key=lambda x: x["try_idx"])
        hit = False
        for t in tries_sorted:
            if t["ok"]:
                first_success[lid] = {"pass1": (t["try_idx"] == 0), "passk": True}
                hit = True
                break
        if not hit:
            first_success[lid] = {"pass1": False, "passk": False}

    total = len(first_success)
    p1 = sum(1 for v in first_success.values() if v["pass1"]) / max(1, total)
    pk = sum(1 for v in first_success.values() if v["passk"]) / max(1, total)

    # stratify using metadata present in rows (propagated from generation)
    def add(acc, key, okv):
        if key is None:
            return
        if key not in acc:
            acc[key] = {"n": 0, "pass1": 0, "passk": 0}
        acc[key]["n"] += 1
        acc[key]["pass1"] += 1 if okv["pass1"] else 0
        acc[key]["passk"] += 1 if okv["passk"] else 0

    by_lang: Dict[str, Dict[str, int]] = {}
    by_cat: Dict[str, Dict[str, int]] = {}
    by_diff: Dict[str, Dict[str, int]] = {}

    # Use any row for its lemma's meta (they all share same meta fields)
    lemma_meta: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r["id"] not in lemma_meta:
            lemma_meta[r["id"]] = {
                "lang": r.get("lang"),
                "category": r.get("category"),
                "difficulty": r.get("difficulty"),
                # 'phenomena' may be added later if needed
            }

    for lid, res in first_success.items():
        meta = lemma_meta.get(lid, {})
        add(by_lang, meta.get("lang"), res)
        add(by_cat, meta.get("category"), res)
        add(by_diff, meta.get("difficulty"), res)

    def finalize(d: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        out = {}
        for k, v in d.items():
            n = max(1, v["n"])
            out[k] = {"n": v["n"], "pass1": v["pass1"] / n, "passk": v["passk"] / n}
        return out

    return {
        "overall": {"n_lemmas": total, "pass1": p1, "passk": pk},
        "by_lang": finalize(by_lang),
        "by_category": finalize(by_cat),
        "by_difficulty": finalize(by_diff),
    }


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
            ok, so, se = check_with_coqc(coq_text, timeout_sec=timeout)
            all_rows.append(
                {
                    "id": lid,
                    "lang": t.get("lang"),
                    "category": t.get("category"),
                    "difficulty": t.get("difficulty"),
                    "ok": ok,
                    "stderr": se,
                    "model": t.get("model"),
                    "try_idx": t["try_idx"],
                    "proof": t["proof"],
                }
            )
            if ok:
                break

    # Write results.jsonl and summary.json in same folder
    mdir = os.path.dirname(proofs_path)
    results_path = os.path.join(mdir, "results.jsonl")
    write_jsonl(results_path, all_rows)
    summary = summarize_passk(all_rows)
    with open(os.path.join(mdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[VER] Wrote {results_path} and summary.json")
    return summary


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
    overall = []
    for mdir in models_dirs:
        proofs_path = os.path.join(mdir, "proofs.jsonl")
        if not os.path.exists(proofs_path):
            print(f"[SKIP] No proofs.jsonl in {mdir}")
            continue
        summary = verify_proofs_file(proofs_path, timeout=args.timeout)
        overall.append({"model": os.path.basename(mdir), **summary})

    # Write overall summary
    overall_path = os.path.join(args.proofs_dir, "overall_summary.json")
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)
    print(f"[VER] Wrote overall summary to {overall_path}")


if __name__ == "__main__":
    main()
