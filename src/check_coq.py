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
            ok = cp.returncode == 0
            return ok, cp.stdout, cp.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"TIMEOUT after {timeout_sec}s"
