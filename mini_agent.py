import json
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

from utils import (
    IGNORED_PATH_NAMES,
    MAX_FILE_VIEW_CHARS,
    MAX_FILE_VIEWS,
    MAX_FILE_VIEWS_TOTAL,
    MAX_HISTORY,
    clip,
    now,
)


class MiniAgent:
    # Mirrors each tool's own default args (see build_tools/validate_tool), so
    # repeated_tool_call() can recognize calls that differ only by omitting a
    # value that matches the default rather than treating them as distinct.
    TOOL_ARG_DEFAULTS: ClassVar[dict] = {
        "list_files": {"path": "."},
        "read_file": {"start": 1, "end": 200},
        "search": {"path": "."},
        "run_shell": {"timeout": 20},
    }

    def __init__(
        self,
        model_client,
        workspace,
        session_store,
        session=None,
        approval_policy="ask",
        max_steps=6,
        max_new_tokens=512,
        depth=0,
        max_depth=1,
        read_only=False,
    ):
        self.model_client = model_client
        self.workspace = workspace
        self.root = Path(workspace.repo_root)
        self.session_store = session_store
        self.approval_policy = approval_policy
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.depth = depth
        self.max_depth = max_depth
        self.read_only = read_only
        self.session = session or {
            "id": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            "created_at": now(),
            "workspace_root": workspace.repo_root,
            "history": [],
            "memory": {"task": "", "files": [], "notes": []},
        }
        self.tools = self.build_tools()
        self.prefix = self.build_prefix()
        self.session_path = self.session_store.save(self.session)

    @classmethod
    def from_session(cls, model_client, workspace, session_store, session_id, **kwargs):
        return cls(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session_store.load(session_id),
            **kwargs,
        )

    @staticmethod
    def remember(bucket, item, limit):
        if not item:
            return
        if item in bucket:
            bucket.remove(item)
        bucket.append(item)
        del bucket[:-limit]

    def build_tools(self):
        tools = {
            "list_files": {
                "schema": {"path": "str='.'"},
                "risky": False,
                "description": "List files in the workspace.",
                "run": self.tool_list_files,
            },
            "read_file": {
                "schema": {"path": "str", "start": "int=1", "end": "int=200"},
                "risky": False,
                "description": "Read a UTF-8 file by line range.",
                "run": self.tool_read_file,
            },
            "search": {
                "schema": {"pattern": "str", "path": "str='.'"},
                "risky": False,
                "description": "Search the workspace with rg or a simple fallback.",
                "run": self.tool_search,
            },
            "run_shell": {
                "schema": {"command": "str", "timeout": "int=20"},
                "risky": True,
                "description": "Run a shell command in the repo root.",
                "run": self.tool_run_shell,
            },
            "write_file": {
                "schema": {"path": "str", "content": "str"},
                "risky": True,
                "description": "Write a text file.",
                "run": self.tool_write_file,
            },
            "patch_file": {
                "schema": {"path": "str", "old_text": "str", "new_text": "str"},
                "risky": True,
                "description": "Replace one exact text block in a file.",
                "run": self.tool_patch_file,
            },
        }
        if self.depth < self.max_depth:
            tools["delegate"] = {
                "schema": {"task": "str", "max_steps": "int=3"},
                "risky": False,
                "description": "Ask a bounded read-only child agent to investigate.",
                "run": self.tool_delegate,
            }
        return tools

    def build_prefix(self):
        tool_lines = []
        for name, tool in self.tools.items():
            fields = ", ".join(f"{key}: {value}" for key, value in tool["schema"].items())
            risk = "approval required" if tool["risky"] else "safe"
            tool_lines.append(f"- {name}({fields}) [{risk}] {tool['description']}")
        tool_text = "\n".join(tool_lines)
        examples = (
            '<tool>{"name":"list_files","args":{"path":"."}}</tool>\n'
            '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>\n'
            '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n'
            "    return -1\n"
            "</content></tool>\n"
            '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>\n'
            '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>\n'
            "<final>Done.</final>"
        )
        rules = (
            "- Use tools instead of guessing about the workspace.\n"
            "- Return exactly one <tool>...</tool> or one <final>...</final>.\n"
            "- Tool calls must look like:\n"
            '  <tool>{"name":"tool_name","args":{...}}</tool>\n'
            "- For write_file and patch_file with multi-line text, prefer XML style:\n"
            '  <tool name="write_file" path="file.py"><content>...</content></tool>\n'
            "- Final answers must look like:\n"
            "  <final>your answer</final>\n"
            "- Never invent tool results.\n"
            "- Keep answers concise and concrete.\n"
            "- If the user asks you to create or update a specific file and the path is clear, "
            "use write_file or patch_file instead of repeatedly listing files.\n"
            "- Before writing tests for existing code, read the implementation first.\n"
            "- When writing tests, match the current implementation unless the user explicitly asked you to change the code.\n"
            "- New files should be complete and runnable, including obvious imports.\n"
            "- Do not repeat the same tool call with the same arguments if it did not help. "
            "Choose a different tool or return a final answer.\n"
            "- Required tool arguments must not be empty. "
            "Do not call read_file, write_file, patch_file, run_shell, or delegate with args={}."
        )
        return "\n\n".join([
            "You are Mini-Coding-Agent, a small local coding agent running through Ollama.",
            "Rules:\n" + rules,
            "Tools:\n" + tool_text,
            "Valid response examples:\n" + examples,
            self.workspace.text(),
        ])

    def memory_text(self):
        memory = self.session["memory"]
        notes = "\n".join(f"- {note}" for note in memory["notes"]) or "- none"
        return "\n".join([
            "Memory:",
            f"- task: {memory['task'] or '-'}",
            f"- files: {', '.join(memory['files']) or '-'}",
            "- notes:",
            notes,
        ])

    def touched_paths(self):
        """Paths the model has successfully read or written, most recent first.

        A write counts as having seen the file: the model authored that content.
        """
        seen = {}
        for item in self.session["history"]:
            if item["role"] != "tool" or item["name"] not in ("read_file", "write_file", "patch_file"):
                continue
            if str(item["content"]).startswith("error"):
                continue
            raw_path = item["args"].get("path")
            if not raw_path:
                continue
            # Re-insert so the dict tracks most-recent order, not first-seen.
            seen.pop(str(raw_path), None)
            seen[str(raw_path)] = int(item["args"].get("start", 1)) if item["name"] == "read_file" else 1
        return list(reversed(list(seen.items())))

    def file_views(self):
        """Re-read touched files from disk so their contents stay in the prompt.

        The transcript clips tool output by recency, which used to drop the very
        file contents a task depends on and left the model re-reading to recover
        them. Reading fresh here also means the view can never go stale after a
        write, and lets the loop guard tell "already have it" from "lost it".
        """
        views = []
        total = 0
        for raw_path, start in self.touched_paths():
            if len(views) >= MAX_FILE_VIEWS or total >= MAX_FILE_VIEWS_TOTAL:
                break
            try:
                path = self.path(raw_path)
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except (OSError, ValueError):  # outside the workspace, gone, or unreadable
                continue
            if not path.is_file():
                continue

            start = max(1, min(start, len(lines) or 1))
            budget = min(MAX_FILE_VIEW_CHARS, MAX_FILE_VIEWS_TOTAL - total)
            body = []
            used = 0
            last = start - 1
            for number, line in enumerate(lines[start - 1:], start=start):
                rendered = f"{number:>4}: {line}"
                if used + len(rendered) + 1 > budget:
                    break
                body.append(rendered)
                used += len(rendered) + 1
                last = number

            views.append({
                "path": str(path.relative_to(self.root)),
                "resolved": str(path),
                "start": start,
                "last": last,
                "total": len(lines),
                "body": "\n".join(body),
                # Partial either because the budget ran out or because the model
                # last read from a later line. Both mean "there is more on disk".
                "partial": last < len(lines) or start > 1,
            })
            total += used
        return views

    def view_for(self, raw_path, views=None):
        if not raw_path:
            return None
        try:
            resolved = str(self.path(raw_path))
        except ValueError:
            return None
        views = self.file_views() if views is None else views
        return next((view for view in views if view["resolved"] == resolved), None)

    def file_views_text(self, views=None):
        views = self.file_views() if views is None else views
        if not views:
            return "Files you have read:\n- none"
        blocks = ["Files you have read (current contents, re-read from disk):"]
        for view in views:
            header = f"- {view['path']} (lines {view['start']}-{view['last']} of {view['total']})"
            if view["partial"]:
                header += " [partial - use read_file with start/end for the rest]"
            blocks.append(header)
            blocks.append(view["body"])
        return "\n".join(blocks)

    def history_text(self, views=None):
        history = self.session["history"]
        if not history:
            return "- empty"

        views = self.file_views() if views is None else views
        lines = []
        seen_reads = set()
        recent_start = max(0, len(history) - 6)
        for index, item in enumerate(history):
            recent = index >= recent_start
            if item["role"] == "tool" and item["name"] in ("write_file", "patch_file"):
                path = str(item["args"].get("path", ""))
                seen_reads.discard(path)

            if item["role"] == "tool" and item["name"] == "read_file":
                view = self.view_for(item["args"].get("path"), views)
                if view is not None:
                    # Content lives in the Files section above; a clipped second
                    # copy here would just crowd out the transcript.
                    lines.append(f"[tool:read_file] {json.dumps(item['args'], sort_keys=True)}")
                    lines.append(f"(contents shown above under {view['path']})")
                    continue
                if not recent:
                    path = str(item["args"].get("path", ""))
                    if path in seen_reads:
                        continue
                    seen_reads.add(path)

            if item["role"] == "tool":
                limit = 900 if recent else 180
                lines.append(f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}")
                lines.append(clip(item["content"], limit))
            else:
                limit = 900 if recent else 220
                lines.append(f"[{item['role']}] {clip(item['content'], limit)}")

        return clip("\n".join(lines), MAX_HISTORY)

    def prompt(self, user_message):
        views = self.file_views()
        return "\n\n".join([
            self.prefix,
            self.memory_text(),
            self.file_views_text(views),
            "Transcript:\n" + self.history_text(views),
            "Current user request:\n" + user_message,
        ])

    def record(self, item):
        self.session["history"].append(item)
        self.session_path = self.session_store.save(self.session)

    def note_tool(self, name, args, result):
        memory = self.session["memory"]
        path = args.get("path")
        if name in {"read_file", "write_file", "patch_file"} and path:
            self.remember(memory["files"], str(path), 8)
        note = f"{name}: {clip(str(result).replace(chr(10), ' '), 220)}"
        self.remember(memory["notes"], note, 5)

    def ask(self, user_message, on_progress=None):
        def notify(message):
            if on_progress:
                on_progress(message)

        memory = self.session["memory"]
        if not memory["task"]:
            memory["task"] = clip(user_message.strip(), 300)
        self.record({"role": "user", "content": user_message, "created_at": now()})

        tool_steps = 0
        attempts = 0
        max_attempts = max(self.max_steps * 3, self.max_steps + 4)

        while tool_steps < self.max_steps and attempts < max_attempts:
            attempts += 1
            notify(f"thinking... (attempt {attempts}/{max_attempts})")
            raw = self.model_client.complete(self.prompt(user_message), self.max_new_tokens)
            kind, payload = self.parse(raw)

            if kind == "tool":
                tool_steps += 1
                name = payload.get("name", "")
                args = payload.get("args", {})
                notify(f"running tool: {name} ({tool_steps}/{self.max_steps})")
                result = self.run_tool(name, args)
                if result.startswith("error"):
                    notify(f"tool {name} failed: {clip(result, 100)}")
                else:
                    notify(f"tool {name} done")
                self.record(
                    {
                        "role": "tool",
                        "name": name,
                        "args": args,
                        "content": result,
                        "created_at": now(),
                    }
                )
                self.note_tool(name, args, result)
                continue

            if kind == "retry":
                notify("model output was malformed, retrying")
                self.record({"role": "assistant", "content": payload, "raw_output": clip(raw), "created_at": now()})
                continue

            final = (payload or raw).strip()
            self.record({"role": "assistant", "content": final, "created_at": now()})
            self.remember(memory["notes"], clip(final, 220), 5)
            return final

        if attempts >= max_attempts and tool_steps < self.max_steps:
            final = (
                "Stopped after too many malformed model responses without a valid tool call or final answer "
                f"(max_steps={self.max_steps}). Increase it with --max-steps if the model needs more attempts."
            )
        else:
            final = (
                f"Stopped after reaching the step limit of {self.max_steps} tool call(s) without a final answer. "
                "Increase it with --max-steps <N> (e.g. --max-steps 12) if the task legitimately needs more tool calls."
            )
        self.record({"role": "assistant", "content": final, "created_at": now()})
        return final

    def run_tool(self, name, args):
        tool = self.tools.get(name)
        if tool is None:
            return f"error: unknown tool '{name}'"
        try:
            self.validate_tool(name, args)
        except Exception as exc:  # noqa: BLE001 - any validation failure becomes a tool-result error, not a crash
            example = self.tool_example(name)
            message = f"error: invalid arguments for {name}: {exc}"
            if example:
                message += f"\nexample: {example}"
            return message
        if self.repeated_tool_call(name, args):
            return f"error: repeated identical tool call for {name}; choose a different tool or return a final answer"
        if tool["risky"] and not self.approve(name, args):
            return f"error: approval denied for {name}"
        try:
            return clip(tool["run"](args))
        except Exception as exc:  # noqa: BLE001 - any tool failure becomes a tool-result error, not a crash
            return f"error: tool {name} failed: {exc}"

    def normalize_args(self, name, args):
        defaults = self.TOOL_ARG_DEFAULTS.get(name, {})
        return {**defaults, **(args or {})}

    def repeated_tool_call(self, name, args):
        tool_events = [item for item in self.session["history"] if item["role"] == "tool"]
        if len(tool_events) < 2:
            return False
        normalized = self.normalize_args(name, args)
        recent = tool_events[-2:]
        repeated = all(
            item["name"] == name and self.normalize_args(item["name"], item["args"]) == normalized
            for item in recent
        )
        if repeated and name == "read_file" and self.view_for(args.get("path")) is None:
            # The contents aren't in the prompt anymore (evicted by the file-view
            # budget, or unreadable), so re-reading is the model's only way back
            # to them. Blocking here is what used to deadlock the loop.
            return False
        return repeated

    def tool_example(self, name):
        examples = {
            "list_files": '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
            "read_file": '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
            "search": '<tool>{"name":"search","args":{"pattern":"binary_search","path":"."}}</tool>',
            "run_shell": '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
            "write_file": '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
            "patch_file": '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
            "delegate": '<tool>{"name":"delegate","args":{"task":"inspect README.md","max_steps":3}}</tool>',
        }
        return examples.get(name, "")

    def validate_tool(self, name, args):
        args = args or {}

        if name == "list_files":
            path = self.path(args.get("path", "."))
            if not path.is_dir():
                raise ValueError("path is not a directory")
            return

        if name == "read_file":
            path = self.path(args["path"])
            if not path.is_file():
                raise ValueError("path is not a file")
            start = int(args.get("start", 1))
            end = int(args.get("end", 200))
            if start < 1 or end < start:
                raise ValueError("invalid line range")
            return

        if name == "search":
            pattern = str(args.get("pattern", "")).strip()
            if not pattern:
                raise ValueError("pattern must not be empty")
            self.path(args.get("path", "."))
            return

        if name == "run_shell":
            command = str(args.get("command", "")).strip()
            if not command:
                raise ValueError("command must not be empty")
            timeout = int(args.get("timeout", 20))
            if timeout < 1 or timeout > 120:
                raise ValueError("timeout must be in [1, 120]")
            return

        if name == "write_file":
            path = self.path(args["path"])
            if path.exists() and path.is_dir():
                raise ValueError("path is a directory")
            if "content" not in args:
                raise ValueError("missing content")
            return

        if name == "patch_file":
            path = self.path(args["path"])
            if not path.is_file():
                raise ValueError("path is not a file")
            old_text = str(args.get("old_text", ""))
            if not old_text:
                raise ValueError("old_text must not be empty")
            if "new_text" not in args:
                raise ValueError("missing new_text")
            text = path.read_text(encoding="utf-8")
            count = text.count(old_text)
            if count != 1:
                raise ValueError(f"old_text must occur exactly once, found {count}")
            return

        if name == "delegate":
            if self.depth >= self.max_depth:
                raise ValueError("delegate depth exceeded")
            task = str(args.get("task", "")).strip()
            if not task:
                raise ValueError("task must not be empty")
            return

    def approve(self, name, args):
        if self.read_only:
            return False
        if self.approval_policy == "auto":
            return True
        if self.approval_policy == "never":
            return False
        try:
            answer = input(f"approve {name} {json.dumps(args, ensure_ascii=True)}? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}

    @staticmethod
    def parse(raw):
        raw = str(raw)
        if "<tool>" in raw and ("<final>" not in raw or raw.find("<tool>") < raw.find("<final>")):
            body = MiniAgent.extract(raw, "tool")
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                return "retry", MiniAgent.retry_notice("model returned malformed tool JSON")
            if not isinstance(payload, dict):
                return "retry", MiniAgent.retry_notice("tool payload must be a JSON object")
            if not str(payload.get("name", "")).strip():
                return "retry", MiniAgent.retry_notice("tool payload is missing a tool name")
            args = payload.get("args", {})
            if args is None:
                payload["args"] = {}
            elif not isinstance(args, dict):
                return "retry", MiniAgent.retry_notice()
            return "tool", payload
        if "<tool" in raw and ("<final>" not in raw or raw.find("<tool") < raw.find("<final>")):
            payload = MiniAgent.parse_xml_tool(raw)
            if payload is not None:
                return "tool", payload
            return "retry", MiniAgent.retry_notice()
        if "<final>" in raw:
            final = MiniAgent.extract(raw, "final").strip()
            if final:
                return "final", final
            return "retry", MiniAgent.retry_notice("model returned an empty <final> answer")
        raw = raw.strip()
        if raw:
            return "final", raw
        return "retry", MiniAgent.retry_notice("model returned an empty response")

    @staticmethod
    def retry_notice(problem=None):
        prefix = "Runtime notice"
        if problem:
            prefix += f": {problem}"
        else:
            prefix += ": model returned malformed tool output"
        return (
            f"{prefix}. Reply with a valid <tool> call or a non-empty <final> answer. "
            'For multi-line files, prefer <tool name="write_file" path="file.py"><content>...</content></tool>.'
        )

    @staticmethod
    def parse_xml_tool(raw):
        match = re.search(r"<tool(?P<attrs>[^>]*)>(?P<body>.*?)</tool>", raw, re.DOTALL)
        if not match:
            return None
        attrs = MiniAgent.parse_attrs(match.group("attrs"))
        name = str(attrs.pop("name", "")).strip()
        if not name:
            return None

        body = match.group("body")
        args = dict(attrs)
        for key in ("content", "old_text", "new_text", "command", "task", "pattern", "path"):
            if f"<{key}>" in body:
                args[key] = MiniAgent.extract_raw(body, key)

        body_text = body.strip("\n")
        if name == "write_file" and "content" not in args and body_text:
            args["content"] = body_text
        if name == "delegate" and "task" not in args and body_text:
            args["task"] = body_text.strip()
        return {"name": name, "args": args}

    @staticmethod
    def parse_attrs(text):
        attrs = {}
        for match in re.finditer(r"""([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""", text):
            attrs[match.group(1)] = match.group(2) if match.group(2) is not None else match.group(3)
        return attrs

    @staticmethod
    def extract(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()

    @staticmethod
    def extract_raw(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:]
        return text[start:end]

    def reset(self):
        self.session["history"] = []
        self.session["memory"] = {"task": "", "files": [], "notes": []}
        self.session_store.save(self.session)

    def path_is_within_root(self, resolved):
        probe = resolved
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        for candidate in (probe, *probe.parents):
            try:
                if candidate.samefile(self.root):
                    return True
            except OSError:
                continue
        return False

    def path(self, raw_path):
        path = Path(raw_path)
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()
        if not self.path_is_within_root(resolved):
            raise ValueError(f"path escapes workspace: {raw_path}")
        return resolved

    def tool_list_files(self, args):
        path = self.path(args.get("path", "."))
        if not path.is_dir():
            raise ValueError("path is not a directory")
        entries = [
            item for item in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
            if item.name not in IGNORED_PATH_NAMES
        ]
        lines = []
        for entry in entries[:200]:
            kind = "[D]" if entry.is_dir() else "[F]"
            lines.append(f"{kind} {entry.relative_to(self.root)}")
        return "\n".join(lines) or "(empty)"

    def tool_read_file(self, args):
        path = self.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        start = int(args.get("start", 1))
        end = int(args.get("end", 200))
        if start < 1 or end < start:
            raise ValueError("invalid line range")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        body = "\n".join(f"{number:>4}: {line}" for number, line in enumerate(lines[start - 1:end], start=start))
        return f"# {path.relative_to(self.root)}\n{body}"

    def tool_search(self, args):
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        path = self.path(args.get("path", "."))

        if shutil.which("rg"):
            result = subprocess.run(
                ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(path)],
                cwd=self.root,
                capture_output=True,
                text=True,
                check=False,
            )
            return result.stdout.strip() or result.stderr.strip() or "(no matches)"

        matches = []
        files = [path] if path.is_file() else [
            item for item in path.rglob("*")
            if item.is_file() and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(self.root).parts)
        ]
        for file_path in files:
            for number, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if pattern.lower() in line.lower():
                    matches.append(f"{file_path.relative_to(self.root)}:{number}:{line}")
                    if len(matches) >= 200:
                        return "\n".join(matches)
        return "\n".join(matches) or "(no matches)"

    def tool_run_shell(self, args):
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command must not be empty")
        timeout = int(args.get("timeout", 20))
        if timeout < 1 or timeout > 120:
            raise ValueError("timeout must be in [1, 120]")
        result = subprocess.run(
            command,
            cwd=self.root,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return "\n".join(
            [
                f"exit_code: {result.returncode}",
                "stdout:",
                result.stdout.strip() or "(empty)",
                "stderr:",
                result.stderr.strip() or "(empty)",
            ]
        )

    def tool_write_file(self, args):
        path = self.path(args["path"])
        content = str(args["content"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {path.relative_to(self.root)} ({len(content)} chars)"

    def tool_patch_file(self, args):
        path = self.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        old_text = str(args.get("old_text", ""))
        if not old_text:
            raise ValueError("old_text must not be empty")
        if "new_text" not in args:
            raise ValueError("missing new_text")
        text = path.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count != 1:
            raise ValueError(f"old_text must occur exactly once, found {count}")
        path.write_text(text.replace(old_text, str(args["new_text"]), 1), encoding="utf-8")
        return f"patched {path.relative_to(self.root)}"

    def tool_delegate(self, args):
        if self.depth >= self.max_depth:
            raise ValueError("delegate depth exceeded")
        task = str(args.get("task", "")).strip()
        if not task:
            raise ValueError("task must not be empty")
        child = MiniAgent(
            model_client=self.model_client,
            workspace=self.workspace,
            session_store=self.session_store,
            approval_policy="never",
            max_steps=int(args.get("max_steps", 3)),
            max_new_tokens=self.max_new_tokens,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            read_only=True,
        )
        child.session["memory"]["task"] = task
        child.session["memory"]["notes"] = [clip(self.history_text(), 300)]
        return "delegate_result:\n" + child.ask(task)
