from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---------------------- Lexicon ----------------------
@dataclass
class Noun:
    key: str  # predicate name in logic/Coq (e.g., Cat)
    en_sg: str
    en_pl: str
    es_sg: str
    es_pl: str
    fr_sg: str
    fr_pl: str


@dataclass
class Name:
    key: str  # constant name in logic/Coq (e.g., garfield)
    surface: Dict[str, str]  # {"en": "Garfield", "es": "Garfield", "fr": "Garfield"}


NOUNS = [
    Noun("Cat", "cat", "cats", "gato", "gatos", "chat", "chats"),
    Noun(
        "Mammal",
        "mammal",
        "mammals",
        "mamífero",
        "mamíferos",
        "mammifère",
        "mammifères",
    ),
    Noun("Reptile", "reptile", "reptiles", "reptil", "reptiles", "reptile", "reptiles"),
    Noun("Bird", "bird", "birds", "ave", "aves", "oiseau", "oiseaux"),
    Noun("Animal", "animal", "animals", "animal", "animales", "animal", "animaux"),
]

NAMES = [
    Name("garfield", {"en": "Garfield", "es": "Garfield", "fr": "Garfield"}),
    Name("tweety", {"en": "Tweety", "es": "Piolín", "fr": "Titi"}),
]

# ------------------- Helper selection ----------------


def choice(seed: random.Random, xs: List):
    return seed.choice(xs)


# ------------------- NL templates --------------------
# (deterministic wording per language; simple morphology)


def nl_all_are(lang: str, A: Noun, B: Noun) -> str:
    if lang == "en":
        return f"All {A.en_pl} are {B.en_pl}."
    if lang == "es":
        return f"Todos los {A.es_pl} son {B.es_pl}."
    if lang == "fr":
        return f"Tous les {A.fr_pl} sont des {B.fr_pl}."
    raise ValueError(lang)


def nl_is_a(lang: str, c: Name, A: Noun) -> str:
    s = c.surface[lang]
    if lang == "en":
        art = "an" if A.en_sg[0].lower() in "aeiou" else "a"
        return f"{s} is {art} {A.en_sg}."
    if lang == "es":
        return f"{s} es un {A.es_sg}."
    if lang == "fr":
        return f"{s} est un {A.fr_sg}."
    raise ValueError(lang)


def nl_is_not_a(lang: str, c: Name, A: Noun) -> str:
    s = c.surface[lang]
    if lang == "en":
        art = "an" if A.en_sg[0].lower() in "aeiou" else "a"
        return f"{s} is not {art} {A.en_sg}."
    if lang == "es":
        return f"{s} no es un {A.es_sg}."
    if lang == "fr":
        return f"{s} n'est pas un {A.fr_sg}."
    raise ValueError(lang)


def nl_no_A_are_B(lang: str, A: Noun, B: Noun) -> str:
    if lang == "en":
        return f"No {A.en_pl} are {B.en_pl}."
    if lang == "es":
        return f"Ningún {A.es_sg} es {B.es_sg}."
    if lang == "fr":
        return f"Aucun {A.fr_sg} n'est {B.fr_sg}."
    raise ValueError(lang)


# -------------------- Logic builders -----------------
# Simple FOL strings (pretty-printed); predicates are uppercase, constants lowercase


def fol_all_are(A: Noun, B: Noun) -> str:
    return f"∀x. {A.key}(x) → {B.key}(x)"


def fol_is_a(c: Name, A: Noun) -> str:
    return f"{A.key}({c.key})"


def fol_no_A_are_B(A: Noun, B: Noun) -> str:
    return f"∀x. {A.key}(x) → ¬{B.key}(x)"


# -------------------- Coq builders -------------------
# We use a minimal semantics setup: Ind : Type, predicates : Ind -> Prop, constants : Ind


def coq_prelude(A: Noun, B: Noun, c: Name) -> List[str]:
    preds = sorted({A.key, B.key})
    lines = [
        "Parameter Ind : Type.",
        *(f"Parameter {p} : Ind -> Prop." for p in preds),
        f"Parameter {c.key} : Ind.",
    ]
    return lines


# Patterns → Coq lemma statements (CIC)


def coq_stmt_univ_inst(A: Noun, B: Noun, c: Name) -> str:
    return (
        "Lemma entailment : "
        f"(forall x:Ind, {A.key} x -> {B.key} x) -> {A.key} {c.key} -> {B.key} {c.key}."
    )


def coq_stmt_negation(A: Noun, B: Noun, c: Name) -> str:
    return (
        "Lemma entailment : "
        f"(forall x:Ind, {A.key} x -> ~ {B.key} x) -> {A.key} {c.key} -> ~ {B.key} {c.key}."
    )


def coq_stmt_transitivity(A: Noun, B: Noun, C: Noun) -> Tuple[str, List[str]]:
    # No named constant here; universal-to-universal
    lines = [
        "Parameter Ind : Type.",
        *(f"Parameter {p} : Ind -> Prop." for p in sorted({A.key, B.key, C.key})),
    ]
    stmt = (
        "Lemma entailment : "
        f"(forall x:Ind, {A.key} x -> {B.key} x) -> "
        f"(forall x:Ind, {B.key} x -> {C.key} x) -> "
        f"(forall x:Ind, {A.key} x -> {C.key} x)."
    )
    return stmt, lines


ALLOWED_TACTICS_SIMPLE = ["intros", "apply", "assumption", "eauto"]
ALLOWED_TACTICS_NEG = [
    "intros",
    "apply",
    "assumption",
    "intro",
    "eauto",
    "contradiction",
]
