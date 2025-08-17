from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List

from .utils import read_jsonl


def summarize_passk(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    rows: flattened per-try verification results including fields:
      id, ok, lang, category, difficulty, phenomena?, model, try_idx
    Computes pass@1, pass@2, pass@3, pass@4, pass@5 by lemma, then aggregates simple stratified stats.
    """
    by_lemma: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_lemma[r["id"]].append(r)

    # Track which attempt succeeded for each lemma
    lemma_results: Dict[str, Dict[str, bool]] = {}
    for lid, tries in by_lemma.items():
        tries_sorted = sorted(tries, key=lambda x: x["try_idx"])

        # Initialize all pass attempts to False
        result = {
            "pass1": False,
            "pass2": False,
            "pass3": False,
            "pass4": False,
            "pass5": False,
        }

        # Find the first successful attempt
        for t in tries_sorted:
            if t["ok"]:
                # Mark the specific attempt as successful
                attempt_num = t["try_idx"] + 1  # try_idx is 0-based, so add 1
                if attempt_num <= 5:  # Only track up to pass5
                    # If we succeed on attempt N, we've also succeeded within N+1, N+2, etc. attempts
                    for i in range(attempt_num, 6):  # 6 because range is exclusive
                        result[f"pass{i}"] = True
                break

        lemma_results[lid] = result

    total = len(lemma_results)

    # Calculate pass rates for each attempt
    p1 = sum(1 for v in lemma_results.values() if v["pass1"]) / max(1, total)
    p2 = sum(1 for v in lemma_results.values() if v["pass2"]) / max(1, total)
    p3 = sum(1 for v in lemma_results.values() if v["pass3"]) / max(1, total)
    p4 = sum(1 for v in lemma_results.values() if v["pass4"]) / max(1, total)
    p5 = sum(1 for v in lemma_results.values() if v["pass5"]) / max(1, total)

    # stratify using metadata present in rows (propagated from generation)
    def add(acc, key, result):
        if key is None:
            return
        if key not in acc:
            acc[key] = {
                "n": 0,
                "pass1": 0,
                "pass2": 0,
                "pass3": 0,
                "pass4": 0,
                "pass5": 0,
            }
        acc[key]["n"] += 1
        acc[key]["pass1"] += 1 if result["pass1"] else 0
        acc[key]["pass2"] += 1 if result["pass2"] else 0
        acc[key]["pass3"] += 1 if result["pass3"] else 0
        acc[key]["pass4"] += 1 if result["pass4"] else 0
        acc[key]["pass5"] += 1 if result["pass5"] else 0

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

    for lid, res in lemma_results.items():
        meta = lemma_meta.get(lid, {})
        add(by_lang, meta.get("lang"), res)
        add(by_cat, meta.get("category"), res)
        add(by_diff, meta.get("difficulty"), res)

    def finalize(d: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        out = {}
        for k, v in d.items():
            n = max(1, v["n"])
            out[k] = {
                "n": v["n"],
                "pass1": v["pass1"] / n,
                "pass2": v["pass2"] / n,
                "pass3": v["pass3"] / n,
                "pass4": v["pass4"] / n,
                "pass5": v["pass5"] / n,
            }
        return out

    return {
        "overall": {
            "n_lemmas": total,
            "pass1": p1,
            "pass2": p2,
            "pass3": p3,
            "pass4": p4,
            "pass5": p5,
        },
        "by_lang": finalize(by_lang),
        "by_category": finalize(by_cat),
        "by_difficulty": finalize(by_diff),
    }


def main():
    """Calculate statistics from verification results."""
    import argparse

    ap = argparse.ArgumentParser(
        description="Calculate statistics from verification results."
    )
    ap.add_argument(
        "--results",
        default="results/batch",
        help="Path to results.jsonl file or directory containing results.",
    )
    ap.add_argument(
        "--out",
        help="Output path for summary (default: summary.json in same directory).",
    )

    args = ap.parse_args()

    if os.path.isfile(args.results):
        # Single file mode
        results = read_jsonl(args.results)
        summary = summarize_passk(results)

        if args.out:
            out_path = args.out
        else:
            out_path = os.path.join(os.path.dirname(args.results), "summary.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Wrote summary to {out_path}")

    elif os.path.isdir(args.results):
        # Directory mode - process all results.jsonl files
        summaries = []
        for root, dirs, files in os.walk(args.results):
            for file in files:
                if file == "results.jsonl":
                    results_path = os.path.join(root, file)
                    model_name = os.path.basename(os.path.dirname(results_path))

                    results = read_jsonl(results_path)
                    summary = summarize_passk(results)
                    summaries.append({"model": model_name, **summary})

                    # Write individual summary
                    summary_path = os.path.join(
                        os.path.dirname(results_path), "summary.json"
                    )
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)
                    print(f"Wrote summary for {model_name} to {summary_path}")

    else:
        print(f"Error: {args.results} is not a file or directory")
        return 1


if __name__ == "__main__":
    main()
