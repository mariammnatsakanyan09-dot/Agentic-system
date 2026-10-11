"""Run the four-role system on a task (for Task 6: give it Task 5).

    export ANTHROPIC_API_KEY=...
    python main.py --task task5.txt --data train.csv test.csv

Everything lands in runs/<timestamp>/: the data files, every script the executor wrote,
transcript.md (who did what) and run.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from agentic.graph import MAX_DECISIONS, build_graph, initial_state
from agentic.tools import CodingTool


def write_report(workdir: Path, final: dict, tool: CodingTool) -> None:
    lines = ["# Run transcript\n"]
    for i, t in enumerate(final["dialogue"], 1):
        lines.append(f"## {i}. {t['role']}\n\n{t['text']}\n")
    lines.append("## Scripts the executor ran\n")
    lines.append("| script | exit code | timed out |\n|---|---|---|")
    for r in tool.runs:
        lines.append(f"| {r.script} | {r.exit_code} | {r.timed_out} |")
    orch = [t for t in final["dialogue"] if t["role"] == "orchestrator"]
    ended = "finish" if orch and orch[-1]["text"].startswith("next: finish") else f"cap of {MAX_DECISIONS} decisions"
    lines.append(f"\nEnded by: **{ended}**  \nTest RMSE printed by a script that ran: **{final['test_rmse']}**")
    (workdir / "transcript.md").write_text("\n".join(lines), encoding="utf-8")
    (workdir / "run.json").write_text(
        json.dumps(
            {"dialogue": final["dialogue"], "decisions": final["decisions"], "test_rmse": final["test_rmse"],
             "runs": [r.to_dict() for r in tool.runs]},
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="text file containing the task (Task 5) word for word")
    ap.add_argument("--data", nargs="*", default=[], help="data files the scripts need")
    ap.add_argument("--model", default=os.environ.get("AGENT_MODEL", "anthropic:claude-sonnet-5-5"))
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    from agentic.llm import LangChainBackend  # imported late so --help works without a provider package

    workdir = Path(args.runs_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    workdir.mkdir(parents=True)
    for f in args.data:
        shutil.copy(f, workdir)

    tool = CodingTool(workdir)
    graph = build_graph(LangChainBackend(args.model), tool)

    task = Path(args.task).read_text(encoding="utf-8")
    names = ", ".join(Path(f).name for f in args.data) or "(none)"
    first = f"{task}\n\nData files in the working directory of every script: {names}"

    printed, final = 0, None
    for state in graph.stream(initial_state(first), stream_mode="values", config={"recursion_limit": 60}):
        for t in state["dialogue"][printed:]:
            print(f"\n=== {t['role']} ===\n{t['text']}\n", flush=True)
        printed, final = len(state["dialogue"]), state

    write_report(workdir, final, tool)
    print(f"Test RMSE printed by a script that ran: {final['test_rmse']}")
    print(f"Report: {workdir / 'transcript.md'}")


if __name__ == "__main__":
    main()
