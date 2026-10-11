import time

from agentic.graph import MAX_DECISIONS, Decision, build_graph, initial_state
from agentic.tools import CodingTool, printed_test_rmse

RMSE_SCRIPT = "print('TRAIN_RMSE_FREE_RUN=0.0100')\nprint('TEST_RMSE_FREE_RUN=0.0123')\n"


def D(nxt):
    return Decision(next=nxt, instruction=f"do {nxt}", reason="because")


class Scripted:
    """Stand-in for the models: scripted decisions, canned speech, real tool calls."""

    def __init__(self, decisions, code=None, summary="done"):
        self.decisions, self.code, self.summary = list(decisions), code, summary
        self.calls = []

    def decide(self, dialogue, *, decisions_left, test_rmse, feedback=None):
        self.calls.append((decisions_left, test_rmse, feedback))
        return self.decisions.pop(0) if len(self.decisions) > 1 else self.decisions[0]

    def speak(self, role, dialogue):
        return f"{role} speaking after {len(dialogue)} turns"

    def execute(self, dialogue, tool):
        if self.code:
            tool.run_python(self.code)
        return self.summary


def run(backend, tmp_path):
    tool = CodingTool(tmp_path)
    final = build_graph(backend, tool).invoke(initial_state("task"), config={"recursion_limit": 60})
    return final, tool


def roles(final):
    return [t["role"] for t in final["dialogue"]]


def test_happy_path(tmp_path):
    b = Scripted([D("scientist"), D("executor"), D("finish")], code=RMSE_SCRIPT)
    final, tool = run(b, tmp_path)
    assert roles(final) == ["user", "orchestrator", "scientist", "orchestrator", "executor", "orchestrator"]
    assert final["test_rmse"] == 0.0123
    assert (tmp_path / "script_01.py").exists()
    assert "TEST_RMSE_FREE_RUN=0.0123" in final["dialogue"][4]["text"]
    assert b.calls[2][1] == 0.0123  # the orchestrator was told the measured value


def test_specialists_read_the_whole_dialogue(tmp_path):
    final, _ = run(Scripted([D("scientist"), D("general"), D("executor"), D("finish")], code=RMSE_SCRIPT), tmp_path)
    assert final["dialogue"][2]["text"] == "scientist speaking after 2 turns"
    assert final["dialogue"][4]["text"] == "general speaking after 4 turns"


def test_finish_refused_without_measured_rmse(tmp_path):
    b = Scripted([D("finish"), D("finish"), D("executor"), D("finish")], code=RMSE_SCRIPT)
    final, _ = run(b, tmp_path)
    assert b.calls[1][2] is not None  # the second ask carried the refusal
    assert roles(final)[:3] == ["user", "orchestrator", "scientist"]  # guard sent it to the scientist
    assert "Guard" in final["dialogue"][1]["text"]
    assert final["test_rmse"] == 0.0123 and roles(final)[-1] == "orchestrator"


def test_rmse_in_model_text_does_not_count(tmp_path):
    b = Scripted([D("executor"), D("finish")], code=None, summary="TEST_RMSE_FREE_RUN=0.001")
    final, _ = run(b, tmp_path)
    assert final["test_rmse"] is None
    assert "ran no script" in final["dialogue"][2]["text"]


def test_failed_script_does_not_count(tmp_path):
    code = "print('TEST_RMSE_FREE_RUN=0.5')\nraise SystemExit(3)\n"
    final, _ = run(Scripted([D("executor"), D("finish")], code=code), tmp_path)
    assert final["test_rmse"] is None


def test_stops_after_eight_decisions(tmp_path):
    final, _ = run(Scripted([D("general")]), tmp_path)
    r = roles(final)
    assert r.count("orchestrator") == MAX_DECISIONS and r.count("general") == MAX_DECISIONS
    assert final["decisions"] == MAX_DECISIONS and r[-1] == "general"


def test_tool_returns_traceback(tmp_path):
    res = CodingTool(tmp_path).run_python("raise ValueError('boom')\n")
    assert res.exit_code == 1 and "ValueError: boom" in res.stderr
    assert printed_test_rmse(res) is None


def test_tool_time_limit(tmp_path):
    t0 = time.time()
    res = CodingTool(tmp_path, time_limit=1).run_python("import time\nprint('hi', flush=True)\ntime.sleep(10)\n")
    assert res.timed_out and res.exit_code == -1 and "hi" in res.stdout and time.time() - t0 < 8


def test_each_call_is_a_new_file(tmp_path):
    t = CodingTool(tmp_path)
    t.run_python("print(1)")
    t.run_python("print(2)")
    assert sorted(p.name for p in tmp_path.glob("script_*.py")) == ["script_01.py", "script_02.py"]
