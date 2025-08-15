# LLM Linguistic Theorem Proving with Coq & Transformers

This project maps natural language entailments to Coq lemmas, prompts an LLM to produce proofs, checks them with Coq, and logs results.

---

## 1. Project layout

```
coq-proof-llm-bench/
    README.md                 # This guide
    requirements.txt          # Python dependencies
    docker/
        Dockerfile            # Docker setup for Coq + Python environment
    data/
        lemmas.jsonl          # Small benchmark of linguistic lemmas
    src/
        schemas.py            # Data classes for lemmas, configs, and results
        prompt.py             # Builds LLM prompts from lemma specs
        generate.py           # Generates proofs (baseline or HF model)
        assemble.py           # Combines lemma + proof into a Coq file
        check_coq.py          # Compiles Coq file to verify proof correctness
        eval.py               # Loads lemmas, runs generation & checking, logs results
        run.py                # CLI entry point for running experiments
        gen_dataset.py        # CLI to generate multilingual JSONL dataset
        ling_grammar.py       # Controlled grammar, lexicon, templates, and logic builders
    results/                  # Stores logs and outputs after runs
```

To generate data:

1. `ling_grammar.py` defines the grammar and lexicon.

2. `gen_dataset.py` generates a JSONL dataset of linguistic entailments.

3. The generated dataset is in `data/`.

To evaluate models:

1. `run.py` is the main entry point → calls `eval.py`.

2. `eval.py` loads lemmas from `data/lemmas.jsonl` using `schemas.py` definitions.

3. For each lemma, `generate.py` produces a proof (either from a baseline or HF model), using `prompt.py` to format the input.

4. The proof is merged with the lemma via `assemble.py` into a `.v` Coq file.

5. `check_coq.py` compiles the `.v` file with `coqc` to check validity.

6. Results are logged in `results/`.

---

## 2. Setup

### Option A: Local (Python + Coq installed)

1. Install **Coq 8.18.x** and ensure `coqc` is on PATH.
2. `python3 -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`

### Option B: Docker (recommended for reproducibility)

1. Install Docker
2. Open a terminal and navigate to the project folder:

```
cd path/to/llm-ling-proof-bench
```

3. Build the Docker image:

```
docker build -t llm-coq:latest docker/
```

This downloads the Coq base image, installs Python, and sets up dependencies.

4. Run the container:

```
docker run --rm -it -v "$PWD":/work -w /work llm-coq:latest bash
```

- `-v "$PWD":/work` mounts your current folder inside the container.

- `-w /work` sets the working directory to the mounted folder.

5. Inside the container, you can now run commands, e.g.:

```
python3 -m src.run --backend baseline -k 1 --limit 2
```

6. Exit the container:

```
exit
```

---

## 3. Evaluate models

1. Baseline run (no LLM, uses included correct proofs)

```
python3 -m src.run --backend baseline -k 1 --limit 2
```

You should see `2/2` lemmas passed.

2. With an HF model:

For dummy tests in local (CPU):

```
python3 -m src.run --backend hf --model distilbert/distilgpt2 -k 2 --limit 2
```

For tests with GPU:

```
python3 -m src.run --backend hf --model Qwen/Qwen2.5-7B-Instruct -k 5 --limit 2
```


---

## 4. Dataset


### 4.1. Dataset description

Each JSON line links NL premises/hypothesis to a Coq lemma. We include two seed items for demonstration:

```jsonl
{"id":"all_cats_mammals","nl_premises":["All cats are mammals.","Garfield is a cat."],"nl_hypothesis":"Garfield is a mammal.","logic_notes":"∀x. Cat x → Mammal x; Cat garfield ⊢ Mammal garfield","coq_prelude":["Parameter Ind : Type.","Parameter Cat Mammal : Ind -> Prop.","Parameter garfield : Ind."],"statement":"Lemma entailment : (forall x:Ind, Cat x -> Mammal x) -> Cat garfield -> Mammal garfield.","category":"quantifiers","phenomena":["universal instantiation","modus ponens"],"allowed_tactics":["intros","apply","assumption","eauto"],"difficulty":"mild","timeout_sec":10,"requires_classical":false,"baseline_proof":"Proof. intros H1 H2. apply H1. exact H2. Qed."}
{"id":"negation_distribution","nl_premises":["No cats are reptiles.","Garfield is a cat."],"nl_hypothesis":"Garfield is not a reptile.","logic_notes":"¬∃x. Cat x ∧ Reptile x; Cat garfield ⊢ ¬Reptile garfield (encoded as Cat x → ¬Reptile x)","coq_prelude":["Parameter Ind : Type.","Parameter Cat Reptile : Ind -> Prop.","Parameter garfield : Ind."],"statement":"Lemma entailment : (forall x:Ind, Cat x -> ~ Reptile x) -> Cat garfield -> ~ Reptile garfield.","category":"negation","phenomena":["universal instantiation","negation"],"allowed_tactics":["intros","apply","assumption","eauto","intro","contradiction"],"difficulty":"mild","timeout_sec":10,"requires_classical":false,"baseline_proof":"Proof. intros Hc Hcat Hrep. apply (Hc garfield); assumption. Qed."}
```

> **Note:** `baseline_proof` is included so the harness can be validated without any LLM.


### 4.2. Dataset generation

Inside your env or Docker container:

```
python3 -m src.gen_dataset \
  --out data/lemmas_auto.jsonl \
  --langs en es fr \
  --n_per_lang 20 \
  --seed 42
````

Review the first 5 lines:

```
awk 'NR<=5 {print}' data/lemmas_auto.jsonl
```

Verify that the dataset is generated correctly by evaluating with baseline:

```
python3 -m src.run --backend baseline --lemmas data/lemmas_auto.jsonl -k 1 --limit 1
```

Then evaluate with HF model using the new JSONL file:

```
python3 -m src.run --backend hf --model distilbert/distilgpt2 -k 2 --limit 5
```
