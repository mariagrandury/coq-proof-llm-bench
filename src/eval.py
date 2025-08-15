import json, time
from typing import Iterable, Dict, Any
from .schemas import LemmaSpec, GenConfig, Result
from .generate import gen_proof
from .assemble import assemble_coq_file
from .check_coq import check_with_coqc

# JSONL utils


def load_lemmas(path: str):
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            yield LemmaSpec(**obj)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# core


def eval_lemma(lemma: LemmaSpec, cfg: GenConfig) -> Iterable[Result]:
    for i in range(cfg.k):
        proof = gen_proof(lemma, cfg, seed=i)
        coq_file = assemble_coq_file(lemma, proof)
        t0 = time.time()
        ok, so, se = check_with_coqc(coq_file, timeout_sec=lemma.timeout_sec)
        dt = int((time.time() - t0) * 1000)
        yield Result(
            id=lemma.id,
            try_idx=i,
            ok=ok,
            ms=dt,
            model=cfg.model_name or "baseline",
            backend=cfg.backend,
            proof=proof,
            stderr=se,
            gen_params={
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "max_new_tokens": cfg.max_new_tokens,
            },
        )
