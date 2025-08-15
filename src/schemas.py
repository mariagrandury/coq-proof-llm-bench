from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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
    lang: Optional[str] = None
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
