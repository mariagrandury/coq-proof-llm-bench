import os
import subprocess
import tempfile
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
            # cp.stderr is a string like this:
            # File "/var/folders/pk/f15jrt057cx1gb096dmkkm900000gn/T/tmpxngfb9es/proof.v", line 8, characters 23-24:
            # Error:
            # Syntax error: [ltac_use_default] expected after [tactic] (in [tactic_command]).
            error = cp.stderr.split("Error:")[1].strip()
            return ok, error
        except subprocess.TimeoutExpired:
            return False, "", f"TIMEOUT after {timeout_sec}s"
