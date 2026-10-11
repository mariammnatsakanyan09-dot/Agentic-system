"""LangGraph wiring for the four-role system (Task 6).

The shared state is the dialogue: a list of turns, each carrying the name of the role that produced it and
the text it said. A node reads every turn already in the state and appends its own. The orchestrator chooses
the next edge; after a specialist runs, an edge returns the turn to the orchestrator.
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .tools import CodingTool, printed_test_rmse

MAX_DECISIONS = 8
SPECIALISTS = ("scientist", "general", "executor")


class Turn(TypedDict):
    role: str  # name of the role that produced the turn
    text: str  # what that role said


class State(TypedDict):
    dialogue: Annotated[list[Turn], operator.add]  # the shared state: every turn so far
    decisions: int  # orchestrator decisions made so far (cap: MAX_DECISIONS)
    next: str  # the edge the orchestrator chose
    test_rmse: Optional[float]  # set ONLY from stdout of a script that really ran


class Decision(BaseModel):
    next: Literal["scientist", "general", "executor", "finish"]
    instruction: str = Field(description="What that role should do now (for finish: closing note for the user).")
    reason: str = Field(description="One sentence: why this role.")


class Backend(Protocol):
    """The three things the graph needs from a model. Real one: agentic.llm.LangChainBackend."""

    def decide(
        self, dialogue: list[Turn], *, decisions_left: int, test_rmse: Optional[float], feedback: Optional[str] = None
    ) -> Decision: ...

    def speak(self, role: str, dialogue: list[Turn]) -> str: ...  # scientist, general

    def execute(self, dialogue: list[Turn], tool: CodingTool) -> str: ...  # executor; returns its summary


def initial_state(task_text: str) -> State:
    return {
        "dialogue": [{"role": "user", "text": task_text}],
        "decisions": 0,
        "next": "",
        "test_rmse": None,
    }


_REFUSED = (
    "Your finish was refused: no executor turn has yet contained a test RMSE printed by a script that actually "
    "ran. Choose scientist, general or executor."
)


def build_graph(backend: Backend, tool: CodingTool):
    def orchestrator(state: State) -> dict:
        used = state["decisions"]
        left = MAX_DECISIONS - used
        d = backend.decide(state["dialogue"], decisions_left=left, test_rmse=state["test_rmse"])

        # Rule: finish only after an executor turn holds a test RMSE that a script really printed.
        if d.next == "finish" and state["test_rmse"] is None:
            d = backend.decide(
                state["dialogue"], decisions_left=left, test_rmse=state["test_rmse"], feedback=_REFUSED
            )
            if d.next == "finish":  # still refused: safety net so the run cannot end unmeasured
                spoken = {t["role"] for t in state["dialogue"]}
                if "scientist" in spoken:
                    d = Decision(
                        next="executor",
                        instruction="Implement the scientist's latest plan in a new script, run it and report "
                        "the free-run RMSE it prints.",
                        reason="Guard: finish refused because no script has printed a test RMSE yet.",
                    )
                else:
                    d = Decision(
                        next="scientist",
                        instruction="Write the differential equation, network, state equation, training "
                        "procedure, free-run test and numbered steps for the problem.",
                        reason="Guard: finish refused because no script has printed a test RMSE yet.",
                    )

        text = f"next: {d.next}\ninstruction: {d.instruction}\nreason: {d.reason}"
        return {
            "dialogue": [{"role": "orchestrator", "text": text}],
            "decisions": used + 1,
            "next": d.next,
        }

    def speaker(role: str):
        def node(state: State) -> dict:
            return {"dialogue": [{"role": role, "text": backend.speak(role, state["dialogue"]).strip()}]}

        return node

    def executor(state: State) -> dict:
        start = len(tool.runs)
        summary = backend.execute(state["dialogue"], tool)
        runs = tool.runs[start:]
        rmse = state["test_rmse"]
        for r in runs:
            got = printed_test_rmse(r)
            if got is not None:
                rmse = got
        parts = [summary.strip()] + [r.as_text() for r in runs]
        if not runs:
            parts.append("[the executor ran no script this turn]")
        return {"dialogue": [{"role": "executor", "text": "\n\n".join(parts)}], "test_rmse": rmse}

    def after_orchestrator(state: State) -> str:
        return END if state["next"] == "finish" else state["next"]

    def after_specialist(state: State) -> str:
        # The edge that returns the turn. The 8th decision's specialist still runs, then the run stops.
        return END if state["decisions"] >= MAX_DECISIONS else "orchestrator"

    g = StateGraph(State)
    g.add_node("orchestrator", orchestrator)
    g.add_node("scientist", speaker("scientist"))
    g.add_node("general", speaker("general"))
    g.add_node("executor", executor)
    g.add_edge(START, "orchestrator")
    g.add_conditional_edges(
        "orchestrator", after_orchestrator, {"scientist": "scientist", "general": "general", "executor": "executor", END: END}
    )
    for role in SPECIALISTS:
        g.add_conditional_edges(role, after_specialist, {"orchestrator": "orchestrator", END: END})
    return g.compile()
