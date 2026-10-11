"""The real backend: every role is a call to a chat model through LangChain."""
from __future__ import annotations

from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool as lc_tool

from . import prompts
from .graph import Decision, Turn
from .tools import CodingTool

MAX_TOOL_ROUNDS = 4  # model/tool exchanges inside one executor turn
DEFAULT_MODEL = "anthropic:claude-sonnet-5-5"


def render(dialogue: list[Turn]) -> str:
    return "\n\n".join(f"[{t['role']}]\n{t['text']}" for t in dialogue)


def _text(msg) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)


class LangChainBackend:
    def __init__(self, model: str = DEFAULT_MODEL):
        # Needs the provider's API key in the environment (e.g. ANTHROPIC_API_KEY).
        self.llm = init_chat_model(model, max_tokens=8000)

    def decide(self, dialogue, *, decisions_left, test_rmse, feedback: Optional[str] = None) -> Decision:
        status = (
            f"STATUS: decisions left including this one: {decisions_left} of 8. "
            f"Test RMSE printed by a script that ran: {test_rmse if test_rmse is not None else 'none yet'}."
        )
        human = render(dialogue) + "\n\n" + status + (f"\n\n{feedback}" if feedback else "")
        model = self.llm.with_structured_output(Decision)
        return model.invoke([SystemMessage(prompts.ORCHESTRATOR), HumanMessage(human)])

    def speak(self, role: str, dialogue) -> str:
        system = {"scientist": prompts.SCIENTIST, "general": prompts.GENERAL}[role]
        return _text(self.llm.invoke([SystemMessage(system), HumanMessage(render(dialogue))]))

    def execute(self, dialogue, tool: CodingTool) -> str:
        @lc_tool
        def run_python(code: str) -> str:
            """Write `code` into a NEW .py file in the working directory and run it (five-minute limit).
            Returns the exit code, standard output and standard error."""
            return tool.run_python(code).as_text()

        model = self.llm.bind_tools([run_python])
        messages = [SystemMessage(prompts.EXECUTOR), HumanMessage(render(dialogue))]
        for _ in range(MAX_TOOL_ROUNDS):
            ai = model.invoke(messages)
            messages.append(ai)
            if not ai.tool_calls:
                return _text(ai)
            for call in ai.tool_calls:
                messages.append(run_python.invoke(call))  # returns a ToolMessage
        messages.append(HumanMessage("The tool budget for this turn is used. Report what you ran and what the script printed."))
        return _text(self.llm.invoke(messages))
