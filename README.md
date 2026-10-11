# Task 6 - Agentic system (LangGraph)

Four roles as nodes of a LangGraph `StateGraph`: **orchestrator**, **scientist**, **general**, **executor**.

## Run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...          

python main.py --task task5.txt --data <data files>
python -m pytest                       
```

`--model` takes any `provider:model` string that LangChain's `init_chat_model` understands
(default `anthropic:claude-sonnet-5-5`; set `AGENT_MODEL` to change it).

Each run writes `runs/<timestamp>/`: the data, every script the executor wrote (`script_01.py`, ...),
`transcript.md` (which agent did what, the scripts and exit codes, the printed test RMSE) and `run.json`.

## How it maps to the spec

| Spec | Where |
|---|---|
| Shared state is the dialogue; each turn = role name + text | `State.dialogue`, `Turn` in `agentic/graph.py` |
| Nodes read all earlier turns and append their own | `speaker()`, `executor()`, `orchestrator()` in `build_graph` |
| Orchestrator picks the next edge; specialists return the turn | conditional edges `after_orchestrator` / `after_specialist` |
| Decision = next role, instruction, one-sentence reason, made by a model call | `Decision` + `LangChainBackend.decide` (structured output). No keyword table, no fixed order |
| Orchestrator writes no theory/code, has no coding tool | it only gets `decide`; no tool is bound to it |
| Stops on `finish` or after 8 decisions | `MAX_DECISIONS = 8` (the 8th decision's specialist still runs, then the run ends) |
| May finish only after an executor turn holds a test RMSE printed by a script that ran | `State.test_rmse` is set only from the stdout of an exit-code-0 script (`printed_test_rmse`); a `finish` without it is refused, re-asked once, then redirected |
| Scientist: equation, network, state equation, training, free-run test, numbered steps; no coding tool | `prompts.SCIENTIST` |
| General: high-level questions only; no coding tool; no fitting plan | `prompts.GENERAL` |
| Only the executor gets the coding tool: write a .py into a working dir, run it with a 5-minute limit, return exit code, stdout, stderr | `CodingTool` in `agentic/tools.py`, bound in `LangChainBackend.execute` |
| Executor returns the printed free-run RMSE, or the traceback; later turns may repair the script, not the plan | `prompts.EXECUTOR`; raw script output is appended to the executor's turn by code, not by the model |

The scripts must print `TRAIN_RMSE_FREE_RUN=<v>` and `TEST_RMSE_FREE_RUN=<v>` (and `TEST_ZERO_BASELINE_RMSE=<v>`);
that fixed marker is how the system tells a measured number from a number someone merely wrote.

## For the solution PDF

Paste/cite `runs/<timestamp>/transcript.md` (which agent did what) and the train/test free-run RMSE from the
executor's script output. Add the link to this repository.

Note: the executor runs model-written code on your machine in the run directory. Use a throwaway environment if that matters.
