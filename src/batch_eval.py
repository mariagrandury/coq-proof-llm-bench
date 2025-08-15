# TODO: add "lang" to results jsonl to be able to aggregate by language
# TODO: more meaningful overall summary, currently just addition of jsonl files

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any, Dict, Iterable, List

from .eval import eval_lemma, load_lemmas
from .schemas import GenConfig

# ----------------------- utils -----------------------


def sanitize_model_dir(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", model_id)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


# ---------------------- stats ------------------------


def _first_success_by_lemma(rows: List[Dict[str, Any]]):
    """Given all tries for a model, return dict lemma_id -> {'passk': bool, 'pass1': bool}.
    Assumes rows contain fields: id, try_idx, ok.
    """
    by_lemma: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_lemma[r["id"]].append(r)
    out = {}
    for lid, tries in by_lemma.items():
        tries_sorted = sorted(tries, key=lambda x: x["try_idx"])  # 0..k-1
        pass1 = bool(tries_sorted and tries_sorted[0]["ok"])  # try 0
        passk = any(t["ok"] for t in tries_sorted)
        out[lid] = {"pass1": pass1, "passk": passk}
    return out


def _group_fields(lemma_meta: Dict[str, Any]) -> Dict[str, Any]:
    # Extract fields we may stratify by; tolerate missing keys
    return {
        "lang": lemma_meta.get("lang"),
        "category": lemma_meta.get("category"),
        "difficulty": lemma_meta.get("difficulty"),
        "phenomena": lemma_meta.get("phenomena", []),  # list
    }


def aggregate_stats(
    all_rows: List[Dict[str, Any]], lemma_index: Dict[str, Dict[str, Any]]
):
    # Overall pass@1, pass@k
    first = _first_success_by_lemma(all_rows)
    total = len(first)
    p1 = sum(1 for v in first.values() if v["pass1"]) / total if total else 0.0
    pk = sum(1 for v in first.values() if v["passk"]) / total if total else 0.0

    # Compile-time (ms) and proof length (chars/lines) over successful first proofs
    ok_rows = [r for r in all_rows if r["ok"]]
    avg_ms = sum(r.get("ms", 0) for r in ok_rows) / max(1, len(ok_rows))

    # very rough length features
    def _lens(s: str):
        return len(s), s.count("\n") + 1

    char_lens = [_lens(r.get("proof", ""))[0] for r in ok_rows]
    line_lens = [_lens(r.get("proof", ""))[1] for r in ok_rows]
    avg_chars = sum(char_lens) / max(1, len(char_lens))
    avg_lines = sum(line_lens) / max(1, len(line_lens))

    # Stratified stats
    by_lang = defaultdict(lambda: {"n": 0, "pass1": 0, "passk": 0})
    by_cat = defaultdict(lambda: {"n": 0, "pass1": 0, "passk": 0})
    by_diff = defaultdict(lambda: {"n": 0, "pass1": 0, "passk": 0})
    by_pheno = defaultdict(lambda: {"n": 0, "pass1": 0, "passk": 0})

    for lid, res in first.items():
        meta = lemma_index.get(lid, {})
        g = _group_fields(meta)

        def upd(d, key):
            if key is None:
                return
            d[key]["n"] += 1
            d[key]["pass1"] += 1 if res["pass1"] else 0
            d[key]["passk"] += 1 if res["passk"] else 0

        upd(by_lang, g.get("lang"))
        upd(by_cat, g.get("category"))
        upd(by_diff, g.get("difficulty"))
        for ph in g.get("phenomena", []) or []:
            upd(by_pheno, ph)

    def finalize(d):
        out = {}
        for k, v in d.items():
            n = max(1, v["n"])
            out[k] = {
                "n": v["n"],
                "pass1": v["pass1"] / n,
                "passk": v["passk"] / n,
            }
        return out

    return {
        "overall": {
            "n_lemmas": total,
            "pass1": p1,
            "passk": pk,
            "avg_compile_ms_success": avg_ms,
            "avg_proof_chars_success": avg_chars,
            "avg_proof_lines_success": avg_lines,
        },
        "by_lang": finalize(by_lang),
        "by_category": finalize(by_cat),
        "by_difficulty": finalize(by_diff),
        "by_phenomena": finalize(by_pheno),
    }


# ---------------------- runner -----------------------


def load_lemma_index(path: str) -> Dict[str, Dict[str, Any]]:
    idx = {}
    import json

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            idx[obj["id"]] = obj
    return idx


def run_for_model(model_id: str, args) -> Dict[str, Any]:
    model_dir = os.path.join(args.outdir, sanitize_model_dir(model_id))
    os.makedirs(model_dir, exist_ok=True)
    raw_path = os.path.join(model_dir, "results.jsonl")

    # Configure generation
    cfg = GenConfig(
        backend="hf",
        model_name=model_id,
        k=args.k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Evaluate
    all_rows: List[Dict[str, Any]] = []
    lemmas = list(load_lemmas(args.lemmas))
    if args.limit:
        lemmas = lemmas[: args.limit]
    for lemma in lemmas:
        for res in eval_lemma(lemma, cfg):
            all_rows.append(asdict(res))
            if res.ok:
                break  # early stop after first success

    write_jsonl(raw_path, all_rows)

    # Aggregate
    lemma_index = load_lemma_index(args.lemmas)
    summary = aggregate_stats(all_rows, lemma_index)
    summary["model_id"] = model_id
    with open(os.path.join(model_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lemmas", default="data/lemmas_auto.jsonl")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--models", nargs="+", help="space-separated list of HF model IDs")
    g.add_argument("--models_file", help="path to file with one HF model ID per line")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--outdir", default="results/batch")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # Collect model IDs
    models: List[str] = []
    if args.models:
        models = args.models
    else:
        with open(args.models_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    models.append(line)

    os.makedirs(args.outdir, exist_ok=True)

    overall = []
    for mid in models:
        print(f"\n=== Evaluating {mid} ===")
        try:
            summary = run_for_model(mid, args)
        except Exception as e:
            summary = {"model_id": mid, "error": str(e)}
        overall.append(summary)

    with open(
        os.path.join(args.outdir, "overall_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    # Print concise table to stdout
    def fmt(x):
        if isinstance(x, float):
            return f"{x*100:.1f}%"
        return str(x)

    print("Model\tPass@1\tPass@k\tN")
    for s in overall:
        if "error" in s:
            print(f"{s['model_id']}	ERROR	-	-")
            continue
        ov = s["overall"]
        print(f"{s['model_id']}	{fmt(ov['pass1'])}	{fmt(ov['passk'])}	{ov['n_lemmas']}")


if __name__ == "__main__":
    main()
