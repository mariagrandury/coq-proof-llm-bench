# LLM Linguistic Theorem Proving – Example Project (Coq + Transformers)

This is a **step‑by‑step, runnable** mini-project tailored for a course on linguistic processing and logical representation. It maps natural language entailments to Coq lemmas, prompts an LLM to produce proofs, checks them with Coq, and logs results.

---

## 0) Project layout

```
llm-ling-proof-bench/
  README.md                 # (this file)
  requirements.txt
  docker/
    Dockerfile
  data/
    lemmas.jsonl            # small seed benchmark
  src/
    schemas.py
    prompt.py
    generate.py
    assemble.py
    check_coq.py
    eval.py
    run.py                  # CLI entry point
  results/
    (created at runtime)
```

---

## 1) Setup

### Option A: Local (Python + Coq installed)

1. Install **Coq 8.17.x** or **8.18.x** and ensure `coqc` is on PATH.
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`

### Option B: Docker (recommended for reproducibility)

```
docker build -t llm-coq:latest docker/
# Run mounting the project
docker run --rm -it -v "$PWD":/work -w /work llm-coq:latest bash
```

---

## 2) Data: `data/lemmas.jsonl`

Each JSON line links NL premises/hypothesis to a Coq lemma. We include two seed items for demonstration:

```jsonl
{"id":"all_cats_mammals","nl_premises":["All cats are mammals.","Garfield is a cat."],"nl_hypothesis":"Garfield is a mammal.","logic_notes":"∀x. Cat x → Mammal x; Cat garfield ⊢ Mammal garfield","coq_prelude":["Parameter Ind : Type.","Parameter Cat Mammal : Ind -> Prop.","Parameter garfield : Ind."],"statement":"Lemma entailment : (forall x:Ind, Cat x -> Mammal x) -> Cat garfield -> Mammal garfield.","category":"quantifiers","phenomena":["universal instantiation","modus ponens"],"allowed_tactics":["intros","apply","assumption","eauto"],"difficulty":"mild","timeout_sec":10,"requires_classical":false,"baseline_proof":"Proof. intros H1 H2. apply H1. exact H2. Qed."}
{"id":"negation_distribution","nl_premises":["No cats are reptiles.","Garfield is a cat."],"nl_hypothesis":"Garfield is not a reptile.","logic_notes":"¬∃x. Cat x ∧ Reptile x; Cat garfield ⊢ ¬Reptile garfield (encoded as Cat x → ¬Reptile x)","coq_prelude":["Parameter Ind : Type.","Parameter Cat Reptile : Ind -> Prop.","Parameter garfield : Ind."],"statement":"Lemma entailment : (forall x:Ind, Cat x -> ~ Reptile x) -> Cat garfield -> ~ Reptile garfield.","category":"negation","phenomena":["universal instantiation","negation"],"allowed_tactics":["intros","apply","assumption","eauto","intro","contradiction"],"difficulty":"mild","timeout_sec":10,"requires_classical":false,"baseline_proof":"Proof. intros Hc Hcat Hrep. apply (Hc garfield); assumption. Qed."}
```

> **Note:** `baseline_proof` is included so the harness can be validated without any LLM.

---

## 3) Python dependencies: `requirements.txt`

```txt
transformers>=4.43
accelerate>=0.33
huggingface_hub>=0.24
pydantic>=2.7
orjson>=3.10
# runtime utils
tqdm>=4.66
```

---

## 4) Dockerfile: `docker/Dockerfile`

```dockerfile
FROM coqorg/coq:8.17.1
# Optional: SerAPI for future interactive checking
RUN opam update && opam install -y coq-serapi || true

# Python
RUN apt-get update && apt-get install -y python3-pip
COPY ../requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
WORKDIR /work
```

---

## 5) Schemas: `src/schemas.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class LemmaSpec:
    id: str
    nl_premises: List[str]
    nl_hypothesis: str
    logic_notes: str
    coq_prelude: List[str]
    statement: str
    category: str
    phenomena: List[str]
    allowed_tactics: List[str]
    difficulty: str
    timeout_sec: int
    requires_classical: bool = False
    baseline_proof: Optional[str] = None

@dataclass
class GenConfig:
    backend: str  # 'baseline' or 'hf'
    model_name: Optional[str] = None
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    k: int = 5

@dataclass
class Result:
    id: str
    try_idx: int
    ok: bool
    ms: int
    model: str
    backend: str
    proof: str
    stderr: str = ""
    gen_params: Optional[Dict[str, Any]] = None
```

---

## 6) Prompting: `src/prompt.py`

```python
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
```

---

## 7) Generation: `src/generate.py`

```python
import re
from typing import Optional
from .schemas import GenConfig, LemmaSpec

# ——— helpers ———
_BLOCK_RE = re.compile(r"Proof\.(.*?)Qed\.", re.S)

def _normalize_to_block(s: str) -> str:
    m = _BLOCK_RE.search(s)
    if m:
        return f"Proof.{m.group(1)}Qed."
    body = s.strip()
    if not body.endswith("Qed."):
        body += "\nQed."
    if not body.startswith("Proof."):
        body = "Proof.\n" + body
    return body

# ——— backends ———

def gen_proof(lemma: LemmaSpec, cfg: GenConfig, seed: int) -> str:
    if cfg.backend == "baseline":
        if lemma.baseline_proof:
            return _normalize_to_block(lemma.baseline_proof)
        return "Proof. Fail trivial. Qed."  # will fail; for testing
    elif cfg.backend == "hf":
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except Exception as e:
            raise RuntimeError("Install transformers to use hf backend") from e
        tok = AutoTokenizer.from_pretrained(cfg.model_name)
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, device_map="auto")
        torch.manual_seed(seed)
        prompt = build_prompt(lemma)  # local import to avoid circular
        # import here to avoid circular dependencies
        input_ids = tok(prompt, return_tensors="pt").to(model.device)
        eos_id = tok.encode("Qed.", add_special_tokens=False)
        eos_id = eos_id[0] if eos_id else None
        out = model.generate(
            **input_ids,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            eos_token_id=eos_id,
        )
        gen = tok.decode(out[0][input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return _normalize_to_block(gen)
    else:
        raise ValueError(f"Unknown backend {cfg.backend}")

# delayed import to avoid cycle
from .prompt import build_prompt
```

---

## 8) Assembly: `src/assemble.py`

```python
from .schemas import LemmaSpec

def assemble_coq_file(lemma: LemmaSpec, proof_block: str) -> str:
    header = "\n".join(lemma.coq_prelude)
    classical = "\nRequire Import Classical." if getattr(lemma, "requires_classical", False) else ""
    return f"""{header}{classical}

{lemma.statement}
{proof_block}
"""
```

---

## 9) Checking with Coq: `src/check_coq.py`

```python
import subprocess, tempfile, os
from typing import Tuple

def check_with_coqc(coq_text: str, timeout_sec: int = 15) -> Tuple[bool, str, str]:
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "proof.v")
        with open(path, "w") as f:
            f.write(coq_text)
        try:
            cp = subprocess.run(
                ["coqc", "-q", path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
                text=True,
            )
            ok = (cp.returncode == 0)
            return ok, cp.stdout, cp.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"TIMEOUT after {timeout_sec}s"
```

---

## 10) Evaluation loop: `src/eval.py`

```python
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
```

---

## 11) CLI: `src/run.py`

```python
import argparse, os, json
from dataclasses import asdict
from collections import defaultdict
from tqdm import tqdm
from .schemas import GenConfig
from .eval import load_lemmas, eval_lemma, write_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lemmas', default='data/lemmas.jsonl')
    ap.add_argument('--backend', choices=['baseline','hf'], default='baseline')
    ap.add_argument('--model', default=None, help='HF model name if backend=hf')
    ap.add_argument('-k', type=int, default=5)
    ap.add_argument('--outdir', default='results')
    ap.add_argument('--max_new_tokens', type=int, default=256)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--limit', type=int, default=0, help='limit number of lemmas (0=all)')
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

    out_path = os.path.join(args.outdir, 'results.jsonl')
    write_jsonl(out_path, all_results)

    # Print simple summary
    total = len(set(r['id'] for r in all_results))
    n_pass = sum(1 for v in passed.values() if v)
    print(f"\nSummary: {n_pass}/{total} lemmas passed at k={args.k} ({100.0*n_pass/max(1,total):.1f}%).")
    print(f"Saved detailed logs to {out_path}")

if __name__ == '__main__':
    main()
```

---

## 12) Run examples

### Dry run (baseline proofs only; no LLM required)

```
python -m src.run --backend baseline -k 1 --limit 2
```

You should see **2/2 lemmas passed** (the baseline includes correct proofs).

### With an HF model (optional)

Provide any chat/instruct causal LLM name (GPU recommended):

```
python -m src.run --backend hf --model Qwen/Qwen2.5-7B-Instruct -k 5 --limit 2
```

---

## 13) Pedagogical extensions

* Add items for **monotonicity** (e.g., encode determiners as generalized quantifiers in a toy way).
* Tag items needing `Require Import Classical.` and compare LLM success vs constructive ones.
* Ablations: remove `logic_notes`; restrict `allowed_tactics`; vary `k`.

---

## 14) Notes

* Coq scripts are brittle across versions—pin via Docker for grading.
* The harness treats **proof checking as ground truth**; any non‑compiling script is marked incorrect.
* Keep preludes lightweight with `Parameter`s so students focus on proof structure (not library details).
