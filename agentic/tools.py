"""The executor's coding tool: write a Python file into the run's working
directory, run it with a five-minute limit, and report what happened."""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

TIME_LIMIT_S = 300  # five minutes, as the assignment requires
_TAIL = 4000  # characters of each stream kept in the dialogue (tail keeps the traceback)

_TEST_RMSE = re.compile(r"^TEST_RMSE_FREE_RUN\s*=\s*([-+0-9.eE]+)\s*$", re.M)


def _text(x) -> str:
    if x is None:
        return ""
    return x.decode("utf-8", "replace") if isinstance(x, bytes) else x


@dataclass
class RunResult:
    script: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    def as_text(self, limit: int = _TAIL) -> str:
        def clip(s: str) -> str:
            return s if len(s) <= limit else "[...clipped...]\n" + s[-limit:]

        flag = " (killed: time limit)" if self.timed_out else ""
        return (
            f"SCRIPT OUTPUT - {self.script}, exit code {self.exit_code}{flag}\n"
            f"[stdout]\n{clip(self.stdout) or '(empty)'}\n"
            f"[stderr]\n{clip(self.stderr) or '(empty)'}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


class CodingTool:
    """Write a Python file into the working directory, run it, return the result."""

    def __init__(self, workdir: Path, time_limit: int = TIME_LIMIT_S):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.time_limit = time_limit
        self.runs: list[RunResult] = []

    def run_python(self, code: str) -> RunResult:
        name = f"script_{len(self.runs) + 1:02d}.py"  # every call is a new file
        (self.workdir / name).write_text(code, encoding="utf-8")
        try:
            p = subprocess.run(
                [sys.executable, name],
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=self.time_limit,
            )
            res = RunResult(name, p.returncode, p.stdout, p.stderr)
        except subprocess.TimeoutExpired as e:
            res = RunResult(
                name, -1, _text(e.stdout),
                _text(e.stderr) + f"\n[killed after {self.time_limit} s]", True,
            )
        self.runs.append(res)
        return res


def printed_test_rmse(res: RunResult) -> Optional[float]:
    """The test RMSE a script that really ran (exit code 0) printed, else None."""
    if res.exit_code != 0:
        return None
    found = _TEST_RMSE.findall(res.stdout)
    if not found:
        return None
    try:
        return float(found[-1])
    except ValueError:
        return None
