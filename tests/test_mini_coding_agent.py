import json
from unittest.mock import patch

import pytest

from fake_model_client import FakeModelClient
from main import build_welcome
from mini_agent import REPEAT_REJECTION, MiniAgent
from ollama_model_client import OllamaModelClient
from session_store import SessionStore
from utils import MAX_FILE_VIEWS
from workspace_context import WorkspaceContext


def build_workspace(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return WorkspaceContext.build(tmp_path)


def build_agent(tmp_path, outputs, **kwargs):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
    approval_policy = kwargs.pop("approval_policy", "auto")
    return MiniAgent(
        model_client=FakeModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy=approval_policy,
        **kwargs,
    )


def test_agent_runs_tool_then_final(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":2}}</tool>',
            "<final>Read the file successfully.</final>",
        ],
    )

    answer = agent.ask("Inspect hello.txt")

    assert answer == "Read the file successfully."
    assert any(item["role"] == "tool" and item["name"] == "read_file" for item in agent.session["history"])
    assert "hello.txt" in agent.session["memory"]["files"]


def test_on_progress_reports_thinking_tool_and_retry(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            "",
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":2}}</tool>',
            "<final>Done.</final>",
        ],
    )

    messages = []
    agent.ask("Inspect hello.txt", on_progress=messages.append)

    assert any("thinking" in message for message in messages)
    assert any("retrying" in message for message in messages)
    assert any("running tool: read_file" in message for message in messages)
    assert any("tool read_file done" in message for message in messages)


def test_agent_retries_after_empty_model_output(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "",
            "<final>Recovered after retry.</final>",
        ],
    )

    answer = agent.ask("Do the task")

    assert answer == "Recovered after retry."
    notices = [item["content"] for item in agent.session["history"] if item["role"] == "assistant"]
    assert any("empty response" in item for item in notices)


def test_agent_retries_after_malformed_tool_payload(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":"bad"}</tool>',
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":1}}</tool>',
            "<final>Recovered after malformed tool output.</final>",
        ],
    )

    answer = agent.ask("Inspect hello.txt")

    assert answer == "Recovered after malformed tool output."
    assert any(item["role"] == "tool" and item["name"] == "read_file" for item in agent.session["history"])
    notices = [item["content"] for item in agent.session["history"] if item["role"] == "assistant"]
    assert any("valid <tool> call" in item for item in notices)
    retries = [item for item in agent.session["history"] if item["role"] == "assistant" and "raw_output" in item]
    assert retries and retries[0]["raw_output"] == '<tool>{"name":"read_file","args":"bad"}</tool>'


def test_agent_accepts_xml_write_file_tool(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="hello.py"><content>print("hi")\n</content></tool>',
            "<final>Done.</final>",
        ],
    )

    answer = agent.ask("Create hello.py")

    assert answer == "Done."
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == 'print("hi")\n'


def test_retries_do_not_consume_the_whole_budget(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "",
            "",
            "<final>Recovered after several retries.</final>",
        ],
        max_steps=1,
    )

    answer = agent.ask("Do the task")

    assert answer == "Recovered after several retries."


def test_agent_saves_and_resumes_session(tmp_path):
    agent = build_agent(tmp_path, ["<final>First pass.</final>"])
    assert agent.ask("Start a session") == "First pass."

    resumed = MiniAgent.from_session(
        model_client=FakeModelClient(["<final>Resumed.</final>"]),
        workspace=agent.workspace,
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.session["history"][0]["content"] == "Start a session"
    assert resumed.ask("Continue") == "Resumed."


def test_delegate_uses_child_agent(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"delegate","args":{"task":"inspect README","max_steps":2}}</tool>',
            "<final>Child result.</final>",
            "<final>Parent incorporated the child result.</final>",
        ],
    )

    answer = agent.ask("Use delegation")

    assert answer == "Parent incorporated the child result."
    tool_events = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert tool_events[0]["name"] == "delegate"
    assert "delegate_result" in tool_events[0]["content"]


def test_patch_file_replaces_exact_match(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool(
        "patch_file",
        {
            "path": "sample.txt",
            "old_text": "world",
            "new_text": "agent",
        },
    )

    assert result == "patched sample.txt"
    assert file_path.read_text(encoding="utf-8") == "hello agent\n"


def test_invalid_risky_tool_does_not_prompt_for_approval(tmp_path):
    agent = build_agent(tmp_path, [], approval_policy="ask")

    with patch("builtins.input") as mock_input:
        result = agent.run_tool("write_file", {})

    assert result.startswith("error: invalid arguments for write_file: 'path'")
    assert 'example: <tool name="write_file"' in result
    mock_input.assert_not_called()


def test_list_files_hides_internal_agent_state(tmp_path):
    agent = build_agent(tmp_path, [])
    (tmp_path / ".mini-coding-agent").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "hello.txt").write_text("hi\n", encoding="utf-8")

    result = agent.run_tool("list_files", {})

    assert ".mini-coding-agent" not in result
    assert ".git" not in result
    assert "[F] hello.txt" in result


def test_path_rejects_parent_escape(tmp_path):
    agent = build_agent(tmp_path, [])

    with pytest.raises(ValueError, match="path escapes workspace"):
        agent.path("../outside.txt")


def test_path_rejects_symlink_escape(tmp_path):
    agent = build_agent(tmp_path, [])
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    link = tmp_path / "outside-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available in this environment")

    with pytest.raises(ValueError, match="path escapes workspace"):
        agent.path("outside-link/secret.txt")


def test_path_accepts_case_variant_on_case_insensitive_filesystems(tmp_path):
    project_root = tmp_path / "Proj"
    project_root.mkdir()
    agent = build_agent(project_root, [])
    variant = project_root.parent / project_root.name.lower() / "README.md"

    if not variant.exists():
        pytest.skip("case-sensitive filesystem")

    resolved = agent.path(str(variant))

    assert resolved.samefile(project_root / "README.md")


def test_repeated_identical_tool_call_is_rejected(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "(empty)", "created_at": "1"})
    agent.record({"role": "tool", "name": "list_files", "args": {}, "content": "(empty)", "created_at": "2"})

    result = agent.run_tool("list_files", {})

    assert result.startswith(REPEAT_REJECTION)
    assert "list_files" in result
    # No file view to point at, so the generic escape hatch still applies.
    assert "Choose a different tool or return a final answer" in result


def test_repeated_tool_call_catches_default_equivalent_args(tmp_path):
    """read_file({"path": x}) and read_file({"path": x, "start": 1, "end": 200}) are the same call."""
    (tmp_path / "snake.html").write_text("<html></html>\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])
    agent.record(
        {"role": "tool", "name": "read_file", "args": {"path": "snake.html"}, "content": "...", "created_at": "1"}
    )
    agent.record(
        {
            "role": "tool",
            "name": "read_file",
            "args": {"path": "snake.html", "start": 1, "end": 200},
            "content": "...",
            "created_at": "2",
        }
    )

    result = agent.run_tool("read_file", {"path": "snake.html"})

    assert result.startswith(REPEAT_REJECTION)


def test_repeat_rejection_names_what_the_model_already_has(tmp_path):
    """The rejection must point at the content, not just refuse the call."""
    (tmp_path / "snake.html").write_text("<html>\n<body>\n</body>\n</html>\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])
    for index in range(2):
        agent.record({
            "role": "tool", "name": "read_file", "args": {"path": "snake.html"},
            "content": "...", "created_at": str(index),
        })

    result = agent.run_tool("read_file", {"path": "snake.html"})

    assert "snake.html" in result
    assert "Files you have read" in result
    assert "patch_file or write_file" in result


def test_repeat_rejection_points_at_the_rest_of_a_partial_file(tmp_path):
    """A truncated view must send the model to a new range, not to a final answer."""
    (tmp_path / "big.py").write_text(_long_file(4000), encoding="utf-8")
    agent = build_agent(tmp_path, [])
    for index in range(2):
        agent.record({
            "role": "tool", "name": "read_file", "args": {"path": "big.py"},
            "content": "...", "created_at": str(index),
        })

    result = agent.run_tool("read_file", {"path": "big.py"})

    assert "start/end" in result
    assert "patch_file or write_file" not in result


def test_stalled_repeat_is_reported_distinctly_from_a_failure(tmp_path):
    """A stall must not look like ordinary tool progress in the REPL."""
    (tmp_path / "notes.txt").write_text("alpha\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"notes.txt"}}</tool>',
            '<tool>{"name":"read_file","args":{"path":"notes.txt"}}</tool>',
            '<tool>{"name":"read_file","args":{"path":"notes.txt"}}</tool>',
            "<final>Done.</final>",
        ],
    )

    messages = []
    agent.ask("look at notes.txt", on_progress=messages.append)

    stalls = [message for message in messages if message.startswith("stalled:")]
    assert stalls, messages
    # Counts consecutive repeats so a deepening stall is visible as it happens.
    assert "1x in a row" in stalls[0]
    assert not any(message.startswith("tool read_file failed") for message in stalls)


def test_welcome_screen_keeps_box_shape_for_long_paths(tmp_path):
    deep = tmp_path / "very" / "long" / "path" / "for" / "the" / "mini" / "agent" / "welcome" / "screen"
    deep.mkdir(parents=True)
    agent = build_agent(deep, [])

    welcome = build_welcome(agent, model="qwen3.5:4b", host="http://127.0.0.1:11434")
    lines = welcome.splitlines()

    assert len(lines) >= 5
    assert len({len(line) for line in lines}) == 1
    assert "..." in welcome
    assert "O   O" in welcome
    assert "MINI-CODING-AGENT" not in welcome
    assert "MINI CODING AGENT" in welcome
    assert "// READY" not in welcome
    assert "SLASH" not in welcome
    assert "READY      " not in welcome
    assert "commands: Commands:" not in welcome


def test_prompt_top_level_sections_stay_flush_left_with_multiline_content(tmp_path):
    workspace = WorkspaceContext(
        cwd=str(tmp_path),
        repo_root=str(tmp_path),
        branch="fix/prompt-indentation",
        default_branch="main",
        status=" M mini_coding_agent.py\n?? tests/test_prompt.py",
        recent_commits=["abc123 first commit", "def456 second commit"],
        project_docs={"README.md": "line1\nline2"},
    )
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
    agent = MiniAgent(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )
    agent.session["memory"] = {
        "task": "verify prompt formatting",
        "files": ["mini_coding_agent.py"],
        "notes": ["saw inconsistent indentation", "need regression coverage"],
    }
    agent.record({"role": "user", "content": "inspect prompt()", "created_at": "1"})
    agent.record(
        {
            "role": "tool",
            "name": "read_file",
            "args": {"path": "mini_coding_agent.py"},
            "content": "    def prompt(self, user_message):\n        ...",
            "created_at": "2",
        }
    )

    prompt = agent.prompt("is this issue legit?")
    lines = prompt.splitlines()

    for label in ["Rules:", "Tools:", "Valid response examples:", "Workspace:", "Memory:", "Transcript:", "Current user request:"]:
        assert label in lines
        assert f"            {label}" not in prompt


def _make_filler(i):
    return {"role": "tool", "name": "list_files", "args": {}, "content": "", "created_at": str(i)}


def test_history_text_deduplicates_reads_but_not_after_write(tmp_path):
    """read_file deduplication must not skip a read that follows a write.

    Realistic prior-turn history (non-recent window):
        user: "update config"
        assistant: <tool>read_file config</tool>
        tool:   config v1 (content: setting=true)
        assistant: <tool>write_file config</tool>
        tool:   wrote
        assistant: <tool>read_file config</tool>
        tool:   config v2 (content: setting=false)   <- MUST NOT be skipped

    Without fix: seen_reads={"config"} after first read; write does NOT clear it;
                 second read is wrongly skipped (LLM sees stale content).
    With fix: write clears seen_reads, second read is correctly shown.
    """
    agent = build_agent(tmp_path, [])

    # Simulate a prior turn with read->write->read on the same file
    # history_length=13, recent_start=7 (indices 0-6 non-recent, 7-12 recent)
    agent.record({"role": "user", "content": "update config", "created_at": "0"})        # index 0
    agent.record({"role": "assistant", "content": '<tool>{"name":"read_file","args":{"path":"config.txt"}}</tool>', "created_at": "1"})
    agent.record({"role": "tool", "name": "read_file", "args": {"path": "config.txt"}, "content": "# config.txt\n   1: setting=true\n", "created_at": "2"})  # index 2, non-recent, ADDED
    agent.record({"role": "assistant", "content": '<tool>{"name":"write_file","args":{"path":"config.txt","content":"setting=false\n"}}</tool>', "created_at": "3"})
    agent.record({"role": "tool", "name": "write_file", "args": {"path": "config.txt", "content": "setting=false\n"}, "content": "wrote config.txt", "created_at": "4"})  # index 4, non-recent
    agent.record({"role": "assistant", "content": '<tool>{"name":"read_file","args":{"path":"config.txt"}}</tool>', "created_at": "5"})
    agent.record({"role": "tool", "name": "read_file", "args": {"path": "config.txt"}, "content": "# config.txt\n   1: setting=false\n", "created_at": "6"})  # index 6, non-recent, ADDED (write cleared dedup)
    # recent entries
    for i in range(7, 13):
        agent.record(_make_filler(i))

    history = agent.history_text()

    # Both read contents appear exactly once (check full line to avoid JSON false positives)
    assert "# config.txt\n   1: setting=true\n" in history
    assert "# config.txt\n   1: setting=false\n" in history
    # Also verify duplicate read (setting=true, same path) does NOT appear twice
    assert history.count("setting=true") == 1


def test_history_text_deduplicates_unchanged_repeated_reads(tmp_path):
    """read_file deduplication should still skip repeated reads with no write in between."""
    agent = build_agent(tmp_path, [])

    # Realistic: two identical reads with no write between them
    # history_length=10, recent_start=4 (indices 0-3 non-recent, 4-9 recent)
    agent.record({"role": "user", "content": "check logs", "created_at": "0"})  # index 0
    agent.record({"role": "assistant", "content": '<tool>{"name":"read_file","args":{"path":"log.txt"}}</tool>', "created_at": "1"})
    agent.record({"role": "tool", "name": "read_file", "args": {"path": "log.txt"}, "content": "# log.txt\n   1: stable\n", "created_at": "2"})  # index 2, non-recent, ADDED
    agent.record({"role": "assistant", "content": '<tool>{"name":"read_file","args":{"path":"log.txt"}}</tool>', "created_at": "3"})  # index 3, non-recent, SKIPPED (dup)
    for i in range(4, 10):
        agent.record(_make_filler(i))  # indices 4-9, recent

    history = agent.history_text()

    # Only first read should appear; duplicates must be skipped
    assert history.count("stable") == 1


def test_ollama_client_posts_expected_payload():
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"response": "<final>ok</final>"}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    client = OllamaModelClient(
        model="qwen3.5:4b",
        host="http://127.0.0.1:11434",
        temperature=0.2,
        top_p=0.9,
        timeout=30,
    )

    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.complete("hello", 42)

    assert result == "<final>ok</final>"
    assert captured["url"] == "http://127.0.0.1:11434/api/generate"
    assert captured["timeout"] == 30
    assert captured["body"]["model"] == "qwen3.5:4b"
    assert captured["body"]["prompt"] == "hello"
    assert captured["body"]["stream"] is False
    assert captured["body"]["raw"] is False
    assert captured["body"]["think"] is False
    assert captured["body"]["options"]["num_predict"] == 42
    assert "num_ctx" not in captured["body"]["options"]


def test_ollama_client_includes_num_ctx_when_context_length_set():
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"response": "<final>ok</final>"}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    client = OllamaModelClient(
        model="qwen3.5:4b",
        host="http://127.0.0.1:11434",
        temperature=0.2,
        top_p=0.9,
        timeout=30,
        context_length=8192,
    )

    with patch("urllib.request.urlopen", fake_urlopen):
        client.complete("hello", 42)

    assert captured["body"]["options"]["num_ctx"] == 8192


def _long_file(lines):
    return "".join(f"line {number} filler filler filler filler filler\n" for number in range(1, lines + 1))


def test_file_view_keeps_contents_visible_as_history_grows(tmp_path):
    """Regression for the re-read deadlock.

    Reproduced from session 20260808-064102-ab9d37: a read_file result was
    clipped by recency until the code the task depended on was no longer in the
    prompt, so the model re-read to recover it and the loop guard blocked it.
    The materialized view must not decay as the transcript grows.
    """
    # The marker sits past the filler, where recency clipping used to cut. It
    # appears nowhere else in the prompt, so finding it can only mean the file
    # view carried it in.
    body = _long_file(120) + "def repair_me():\n    return 'TAIL_MARKER'\n"
    (tmp_path / "app.py").write_text(body, encoding="utf-8")
    agent = build_agent(tmp_path, [])

    agent.record({"role": "user", "content": "fix the bug", "created_at": "0"})
    agent.record({
        "role": "tool", "name": "read_file", "args": {"path": "app.py"},
        "content": "# app.py\n   1: line 1 filler\n", "created_at": "1",
    })
    assert "TAIL_MARKER" in agent.prompt("fix the bug")

    # The tail must survive many more turns, not erode out as history grows.
    for index in range(2, 20):
        agent.record(_make_filler(index))
    assert "TAIL_MARKER" in agent.prompt("fix the bug")
    # Guard the test itself: the transcript alone must not supply the marker,
    # otherwise this would pass even with the file view removed.
    assert "TAIL_MARKER" not in agent.history_text(views=[])


def test_file_view_reflects_disk_after_write(tmp_path):
    """A write must not leave a stale materialized copy behind."""
    (tmp_path / "conf.txt") .write_text("setting=true\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])
    agent.record({
        "role": "tool", "name": "read_file", "args": {"path": "conf.txt"},
        "content": "# conf.txt\n   1: setting=true\n", "created_at": "0",
    })
    assert "setting=true" in agent.file_views_text()

    (tmp_path / "conf.txt").write_text("setting=false\n", encoding="utf-8")
    views = agent.file_views_text()
    assert "setting=false" in views
    assert "setting=true" not in views


def test_repeated_read_allowed_when_view_is_unavailable(tmp_path):
    """The guard may only block a re-read the harness can still satisfy."""
    (tmp_path / "kept.txt").write_text("kept\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])
    for index in range(2):
        agent.record({
            "role": "tool", "name": "read_file", "args": {"path": "kept.txt"},
            "content": "# kept.txt\n   1: kept\n", "created_at": str(index),
        })

    # Materialized -> the contents are in the prompt, so the repeat is redundant.
    assert agent.repeated_tool_call("read_file", {"path": "kept.txt"}) is True

    # Not materialized (deleted from disk) -> blocking would strand the model.
    (tmp_path / "kept.txt").unlink()
    assert agent.repeated_tool_call("read_file", {"path": "kept.txt"}) is False


def test_file_view_budget_evicts_oldest_and_marks_partial(tmp_path):
    """Budgets are enforced, and a clipped view says so."""
    for name in ("a.py", "b.py", "c.py", "d.py"):
        (tmp_path / name).write_text(f"# {name}\n", encoding="utf-8")
    (tmp_path / "big.py").write_text(_long_file(4000), encoding="utf-8")
    agent = build_agent(tmp_path, [])
    for index, name in enumerate(("a.py", "b.py", "c.py", "d.py")):
        agent.record({
            "role": "tool", "name": "read_file", "args": {"path": name},
            "content": f"# {name}\n", "created_at": str(index),
        })

    views = agent.file_views()
    assert len(views) == MAX_FILE_VIEWS
    # Most recent kept, oldest evicted.
    assert [view["path"] for view in views] == ["d.py", "c.py", "b.py"]

    agent.record({
        "role": "tool", "name": "read_file", "args": {"path": "big.py"},
        "content": "# big.py\n", "created_at": "9",
    })
    big = agent.view_for("big.py")
    assert big["partial"] is True
    assert big["last"] < big["total"]
    assert "partial" in agent.file_views_text()


def test_materialized_read_collapses_to_pointer_in_transcript(tmp_path):
    """Transcript keeps a reference, not a second clipped copy of the bytes."""
    (tmp_path / "app.py").write_text("alpha\nbravo\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])
    agent.record({
        "role": "tool", "name": "read_file", "args": {"path": "app.py"},
        "content": "# app.py\n   1: alpha\n   2: bravo\n", "created_at": "0",
    })
    history = agent.history_text()
    assert "contents shown above under app.py" in history
    assert "1: alpha" not in history
    assert "1: alpha" in agent.file_views_text()


def test_stuck_read_loop_from_session_20260808_064102_is_recoverable(tmp_path):
    """End-to-end regression for the deadlock in session 20260808-064102-ab9d37.

    Original shape: write a file, be told it has a bug, read it, then repeat that
    same read until --max-steps ran out -- because the contents had been clipped
    out of the prompt and the guard refused the re-read that would restore them.

    Now the contents stay in the prompt, so the model can act on the second turn
    instead of circling, and the run ends with the file actually fixed.
    """
    body = _long_file(120) + "function gameLoop() {\n  head = snake[0] + BUGGY_DELTA;\n}\n"
    (tmp_path / "snake.html").write_text(body, encoding="utf-8")

    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"snake.html","start":1,"end":200}}</tool>',
            (
                '<tool name="patch_file" path="snake.html">'
                "<old_text>head = snake[0] + BUGGY_DELTA;</old_text>"
                "<new_text>head = snake[0] + FIXED_DELTA;</new_text></tool>"
            ),
            "<final>Fixed the immediate game over.</final>",
        ],
        max_steps=10,
    )

    messages = []
    answer = agent.ask("the game ends immediately, please fix it", on_progress=messages.append)

    assert answer == "Fixed the immediate game over."
    assert "FIXED_DELTA" in (tmp_path / "snake.html").read_text(encoding="utf-8")
    # The buggy line had to be visible for the patch to be possible at all.
    assert "BUGGY_DELTA" in agent.model_client.prompts[1]
    # Nowhere near the step limit, and no stall.
    tool_events = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert len(tool_events) == 2
    assert not [message for message in messages if message.startswith("stalled:")]


def test_stubborn_repeat_loop_terminates_within_budget(tmp_path):
    """Even a model that will not move on must end the turn, visibly."""
    (tmp_path / "snake.html").write_text(_long_file(10), encoding="utf-8")
    repeat = '<tool>{"name":"read_file","args":{"path":"snake.html","start":1,"end":200}}</tool>'
    agent = build_agent(tmp_path, [repeat] * 12, max_steps=4)

    messages = []
    answer = agent.ask("fix it", on_progress=messages.append)

    assert "step limit" in answer
    tool_events = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert len(tool_events) == agent.max_steps
    # The guard compares against the last two tool events, so the first two
    # reads are served before any repeat can be established.
    stalls = [message for message in messages if message.startswith("stalled:")]
    assert len(stalls) == agent.max_steps - 2
    assert "2x in a row" in stalls[-1]
    # The rejection stays actionable rather than repeating a generic refusal.
    assert all("Files you have read" in item["content"] for item in tool_events[2:])
