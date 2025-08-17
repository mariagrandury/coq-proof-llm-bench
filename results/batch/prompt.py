from typing import List

from .schemas import LemmaSpec


def build_prompt(lemma: LemmaSpec) -> str:
    imports_block = "\n".join(lemma.coq_prelude)
    tactics = ", ".join(lemma.allowed_tactics)
    phenomena = ", ".join(lemma.phenomena)
    nl_block = "\n".join([f"- {p}" for p in lemma.nl_premises])
    return f"""
You are a Coq assistant verifying a linguistic entailment.
Phenomena: {phenomena}.
Only output Coq between 'Proof.' and 'Qed.'. No comments, no explanations.

Allowed tactics: {tactics}.

Prelude:
{imports_block}

Lemma:
{lemma.statement}

(Informal gloss)
Premises:\n{nl_block}\nHypothesis: {lemma.nl_hypothesis}
Logic notes: {lemma.logic_notes}

Now output:
Proof.
""".strip()
