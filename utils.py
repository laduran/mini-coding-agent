import shutil
from datetime import datetime, timezone

DOC_NAMES = ("AGENTS.md", "README.md", "pyproject.toml", "package.json")
HELP_TEXT = "/help, /memory, /session, /reset, /exit"
WELCOME_ART = (
    "/\\     /\\\\",
    "{  `---'  }",
    "{  O   O  }",
    "~~>  V  <~~",
    "\\\\  \\|/  /",
    "`-----'__",
)
HELP_DETAILS = "\n".join(
    [
        "Commands:",
        "/help    Show this help message.",
        "/memory  Show the agent's distilled working memory.",
        "/session Show the path to the saved session file.",
        "/reset   Clear the current session history and memory.",
        "/exit    Exit the agent.",
    ]
)
MAX_TOOL_OUTPUT = 4000
MAX_HISTORY = 12000
IGNORED_PATH_NAMES = {".git", ".mini-coding-agent", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv"}


def now():
    return datetime.now(timezone.utc).isoformat()


def clip(text, limit=MAX_TOOL_OUTPUT):
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def middle(text, limit):
    text = str(text).replace("\n", " ")
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    left = (limit - 3) // 2
    right = limit - 3 - left
    return text[:left] + "..." + text[-right:]
