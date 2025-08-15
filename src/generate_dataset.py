from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List

from .ling_grammar import (
    ALLOWED_TACTICS_NEG,
    ALLOWED_TACTICS_SIMPLE,
    NAMES,
    NOUNS,
    Name,
    Noun,
    choice,
    coq_prelude,
    coq_stmt_negation,
    coq_stmt_transitivity,
    coq_stmt_univ_inst,
    fol_all_are,
    fol_is_a,
    fol_no_A_are_B,
    nl_all_are,
    nl_is_a,
    nl_is_not_a,
    nl_no_A_are_B,
)

LANGS = ["en", "es", "fr"]


def make_univ_inst(seed: random.Random, lang: str) -> Dict:
    A, B = seed.sample(NOUNS, 2)
    c = choice(seed, NAMES)
    premises = [nl_all_are(lang, A, B), nl_is_a(lang, c, A)]
    hypothesis = nl_is_a(lang, c, B)
    logic = {
        "premises": [fol_all_are(A, B), fol_is_a(c, A)],
        "hypothesis": fol_is_a(c, B),
    }
    prelude = coq_prelude(A, B, c)
    statement = coq_stmt_univ_inst(A, B, c)
    return {
        "id": f"univ_inst__{lang}__{A.key}_{B.key}__{c.key}",
        "lang": lang,
        "nl_premises": premises,
        "nl_hypothesis": hypothesis,
        "logic_notes": f"{fol_all_are(A,B)}; {fol_is_a(c,A)} ⊢ {fol_is_a(c,B)}",
        "coq_prelude": prelude,
        "statement": statement,
        "category": "quantifiers",
        "phenomena": ["universal instantiation", "modus ponens"],
        "allowed_tactics": ALLOWED_TACTICS_SIMPLE,
        "difficulty": "mild",
        "timeout_sec": 10,
        "requires_classical": False,
        "baseline_proof": "Proof. intros H1 H2. apply H1. exact H2. Qed.",
    }


def make_negation(seed: random.Random, lang: str) -> Dict:
    A, B = seed.sample(NOUNS, 2)
    c = choice(seed, NAMES)
    premises = [nl_no_A_are_B(lang, A, B), nl_is_a(lang, c, A)]
    hypothesis = nl_is_not_a(lang, c, B)
    logic = {
        "premises": [fol_no_A_are_B(A, B), fol_is_a(c, A)],
        "hypothesis": f"¬{B.key}({c.key})",
    }
    prelude = coq_prelude(A, B, c)
    statement = coq_stmt_negation(A, B, c)
    return {
        "id": f"negation__{lang}__{A.key}_{B.key}__{c.key}",
        "lang": lang,
        "nl_premises": premises,
        "nl_hypothesis": hypothesis,
        "logic_notes": f"{fol_no_A_are_B(A,B)}; {fol_is_a(c,A)} ⊢ ¬{B.key}({c.key})",
        "coq_prelude": prelude,
        "statement": statement,
        "category": "negation",
        "phenomena": ["universal implication", "negation"],
        "allowed_tactics": ALLOWED_TACTICS_NEG,
        "difficulty": "mild",
        "timeout_sec": 10,
        "requires_classical": False,
        "baseline_proof": "Proof. intros Hc Hcat Hrep. apply (Hc garfield); assumption. Qed.",
    }


def make_transitivity(seed: random.Random, lang: str) -> Dict:
    A, B, C = seed.sample(NOUNS, 3)
    # Use a neutral English gloss in NL section? We keep NL empty here since it's schema-agnostic,
    # or provide a meta description (optional). We'll synthesize pseudo-NL for each lang.
    if lang == "en":
        premises = [f"All {A.en_pl} are {B.en_pl}.", f"All {B.en_pl} are {C.en_pl}."]
        hypothesis = f"All {A.en_pl} are {C.en_pl}."
    elif lang == "es":
        premises = [
            f"Todos los {A.es_pl} son {B.es_pl}.",
            f"Todos los {B.es_pl} son {C.es_pl}.",
        ]
        hypothesis = f"Todos los {A.es_pl} son {C.es_pl}."
    elif lang == "fr":
        premises = [
            f"Tous les {A.fr_pl} sont des {B.fr_pl}.",
            f"Tous les {B.fr_pl} sont des {C.fr_pl}.",
        ]
        hypothesis = f"Tous les {A.fr_pl} sont des {C.fr_pl}."
    else:
        raise ValueError(lang)

    logic_notes = f"{fol_all_are(A,B)}; {fol_all_are(B,C)} ⊢ {fol_all_are(A,C)}"
    stmt, prel = coq_stmt_transitivity(A, B, C)
    return {
        "id": f"transitivity__{lang}__{A.key}_{B.key}_{C.key}",
        "lang": lang,
        "nl_premises": premises,
        "nl_hypothesis": hypothesis,
        "logic_notes": logic_notes,
        "coq_prelude": prel,
        "statement": stmt,
        "category": "quantifiers",
        "phenomena": ["transitivity", "universal reasoning"],
        "allowed_tactics": ALLOWED_TACTICS_SIMPLE,
        "difficulty": "mild",
        "timeout_sec": 10,
        "requires_classical": False,
        "baseline_proof": "Proof. intros H1 H2 x Hx. apply H2, H1, Hx. Qed.",
    }


PATTERNS = [make_univ_inst, make_negation, make_transitivity]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/lemmas_auto.jsonl")
    ap.add_argument("--langs", nargs="+", default=["en", "es", "fr"], choices=LANGS)
    ap.add_argument(
        "--n_per_lang",
        type=int,
        default=30,
        help="approx. samples per language (uniform across patterns)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    per_pattern = max(1, args.n_per_lang // len(PATTERNS))

    rows: List[Dict] = []
    for lang in args.langs:
        for pat in PATTERNS:
            for _ in range(per_pattern):
                rows.append(pat(rnd, lang))

    # Deduplicate IDs if accidental collisions arise
    seen = set()
    out_rows = []
    for r in rows:
        i = r["id"]
        k = 1
        while r["id"] in seen:
            r["id"] = f"{i}__{k}"
            k += 1
        seen.add(r["id"])
        out_rows.append(r)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_rows)} examples to {args.out}")


if __name__ == "__main__":
    main()
