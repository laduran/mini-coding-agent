# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A minimal, standalone local coding agent (fork of Sebastian Raschka's `mini-coding-agent`) backed by Ollama. Zero runtime dependencies beyond the stdlib — `python main.py` works with no install. The design intentionally favors readability over robustness (see README "Notes & Tips").

## Commands

```bash
# Run (uv, preferred)
uv run mini-coding-agent [--approval ask|auto|never] [--model NAME] [--cwd PATH]

# Run without uv
python main.py

# One-shot prompt instead of REPL
uv run mini-coding-agent "write a function that..."

# Tests
uv run python -m pytest -q
uv run python -m pytest -q tests/test_mini_coding_agent.py::test_agent_runs_tool_then_final  # single test

# Lint
uv run python -m ruff check .
```

CI (`.github/workflows/*.yml`) runs this same lint+test combo across ubuntu/macos/windows, Python 3.10, via both `pip install -e .` and `uv sync --group dev`. There is no separate build/typecheck step.

Requires a running Ollama server (`ollama serve`) with a pulled model (default `qwen3.5:9b`) for real usage; tests use `FakeModelClient` (`fake_model_client.py`) instead and don't need Ollama running.

## Architecture

Everything is flat, top-level `.py` modules (no package dir) — see `[tool.setuptools] py-modules` in `pyproject.toml` for the authoritative module list.

**`main.py`** — CLI entry point (`mini-coding-agent` script). Parses args, builds `WorkspaceContext` + `OllamaModelClient` + `SessionStore`, constructs a `MiniAgent` (fresh or resumed via `--resume`), and runs either a one-shot prompt or the REPL loop. Slash commands (`/help`, `/memory`, `/session`, `/reset`, `/exit`) are intercepted in the REPL loop itself, not routed through the agent/model.

**`mini_agent.py`** — the whole harness lives in `MiniAgent`. Key pieces, in the order a request flows through them:

1. **Prefix / prompt assembly** (`build_prefix`, `prompt`): a stable instruction+tool-schema prefix (built once at construction, cache-friendly) is concatenated each turn with distilled memory, clipped transcript history, and the current user message. There is no chat/message-array API — the whole thing is one text blob POSTed to Ollama's `/api/generate`.
2. **Tool protocol** (`build_tools`, `parse`, `parse_xml_tool`): the model must reply with exactly one `<tool>{"name":...,"args":{...}}</tool>` (JSON style) or `<tool name="..." path="..."><content>...</content></tool>` (XML style, for multi-line file content) or `<final>...</final>`. `parse()` picks JSON vs XML vs final vs "retry" (malformed output produces a `retry_notice` fed back to the model rather than crashing). This custom protocol — not native Ollama tool-calling — is a known friction point (see `TEST_PROMPTS.md` notes comparing against Open WebUI).
3. **Tool execution** (`run_tool`, `validate_tool`, `tool_*` methods): tools are `list_files`, `read_file`, `search`, `run_shell`, `write_file`, `patch_file`, and (if `depth < max_depth`) `delegate`. Each tool is validated before running; risky tools (`run_shell`, `write_file`, `patch_file`) go through `approve()`, gated by `--approval ask|auto|never`. All filesystem access goes through `path()`, which resolves and confirms the target is inside `workspace.repo_root` (`path_is_within_root`) — this is the sandboxing boundary, don't bypass it when adding new file-touching tools.
4. **Loop control** (`ask`): iterates up to `max_steps` tool calls (with a separate, larger `max_attempts` budget that also counts malformed-output retries) until the model returns `<final>`. `repeated_tool_call()` rejects a tool call that's identical (args normalized against `TOOL_ARG_DEFAULTS`) to either of the last two tool calls, to break the model out of stall loops.
5. **Memory & transcript** (`note_tool`, `memory_text`, `history_text`, `record`): every turn is appended to `session["history"]` and persisted immediately via `SessionStore.save`. `session["memory"]` (task/files/notes, each capped via `remember()`) is a small distilled summary threaded into every prompt so context survives beyond the clipped transcript. `history_text()` deduplicates repeated `read_file` results for the same path (dropped if unchanged since a later write) to control prompt growth, and gives recent turns (last 6) a bigger clip budget than older ones.
6. **Delegation** (`tool_delegate`): spawns a child `MiniAgent` at `depth+1`, `read_only=True`, `approval_policy="never"`, sharing the same model client/workspace/session_store — bounded (`max_depth`, default 1) so delegation can't recurse indefinitely.

**`workspace_context.py`** — `WorkspaceContext.build()` snapshots repo facts once at startup (git root, branch, status, recent commits, and the contents of any `AGENTS.md`/`README.md`/`pyproject.toml`/`package.json` found — see `DOC_NAMES` in `utils.py`) into `.text()`, which becomes part of the stable prefix. Git calls fail soft (empty/fallback values) if git is unavailable.

**`ollama_model_client.py`** — thin wrapper posting to `{host}/api/generate` with `stream: false`. `FakeModelClient` (`fake_model_client.py`) is the test double: takes a queue of canned outputs and records prompts it was called with.

**`session_store.py`** — sessions are plain JSON files under `<repo_root>/.mini-coding-agent/sessions/<session_id>.json`, written on every `record()` call (not just on exit), so `--resume latest` / `--resume <id>` can pick up mid-session.

## Notes for making changes

- New tools: add an entry to `build_tools()`, a `tool_example()` entry, validation in `validate_tool()`, and if it touches the filesystem, route paths through `self.path()`. Add the default-args mapping to `TOOL_ARG_DEFAULTS` if the tool has optional args that should count as "the same call" when omitted.
- The model only ever sees the *text* the prefix/prompt building produces — if you change prompt structure, check `tests/test_mini_coding_agent.py` (several tests assert exact prompt layout, e.g. `test_prompt_top_level_sections_stay_flush_left_with_multiline_content`).
- `TEST_PROMPTS.md` tracks real-model manual test prompts (not run in CI) for comparing harness/protocol behavior across models — add to it when a prompt surfaces a harness bug worth re-checking later.
