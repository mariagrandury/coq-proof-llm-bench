from .schemas import LemmaSpec


def assemble_coq_file(lemma: LemmaSpec, proof_block: str) -> str:
    header = "\n".join(lemma.coq_prelude)
    classical = (
        "\nRequire Import Classical."
        if getattr(lemma, "requires_classical", False)
        else ""
    )
    return f"""{header}{classical}

{lemma.statement}
{proof_block}
"""
