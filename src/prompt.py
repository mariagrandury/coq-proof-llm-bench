from typing import List

from .schemas import LemmaSpec


def build_prompt(lemma: LemmaSpec) -> str:
    imports_block = "\n".join(lemma.coq_prelude)
    tactics = ", ".join(lemma.allowed_tactics)
    phenomena = ", ".join(lemma.phenomena)
    nl_block = "\n".join([f"- {p}" for p in lemma.nl_premises])
    return f"""
You are a Coq assistant verifying a linguistic entailment.

Your task is to produce a single valid Coq proof script for the lemma below. Start the proof after 'Proof.' and end it with 'Qed.'. Do not restate the lemma and do not include explanations. Ensure the script type-checks end-to-end under Coq's kernel.

Phenomena: {phenomena}.
Allowed tactics: {tactics}.

Prelude:
{imports_block}

Lemma to prove:
{lemma.statement}

(Informal gloss)
Premises:
{nl_block}
Hypothesis: {lemma.nl_hypothesis}
Logic notes: {lemma.logic_notes}

Proof without additional comments:
Proof.
""".strip()
