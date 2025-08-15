import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict

from tqdm import tqdm

from .eval import eval_lemma, load_lemmas, write_jsonl
from .schemas import GenConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lemmas", default="data/lemmas.jsonl")
    ap.add_argument("--backend", choices=["baseline", "hf"], default="baseline")
    ap.add_argument("--model", default=None, help="HF model name if backend=hf")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument(
        "--limit", type=int, default=0, help="limit number of lemmas (0=all)"
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = GenConfig(
        backend=args.backend,
        model_name=args.model,
        k=args.k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    all_results = []
    passed = defaultdict(lambda: False)

    for idx, lemma in enumerate(load_lemmas(args.lemmas)):
        if args.limit and idx >= args.limit:
            break
        for res in tqdm(eval_lemma(lemma, cfg), desc=f"{lemma.id}"):
            all_results.append(asdict(res))
            if res.ok:
                passed[lemma.id] = True
                break  # early stop on first success

    out_path = os.path.join(args.outdir, "results.jsonl")
    write_jsonl(out_path, all_results)

    # Print simple summary
    total = len(set(r["id"] for r in all_results))
    n_pass = sum(1 for v in passed.values() if v)
    print(
        f"\nSummary: {n_pass}/{total} lemmas passed at k={args.k} ({100.0*n_pass/max(1,total):.1f}%)."
    )
    print(f"Saved detailed logs to {out_path}")


if __name__ == "__main__":
    main()
