import os
import re
import sys
from datetime import datetime, timezone

DOC_NAMES = ("AGENTS.md", "README.md", "pyproject.toml", "package.json")
HELP_TEXT = "/help, /memory, /session, /reset, /exit"
WELCOME_ART = (
    "                 _~                     ",
    "               ~~~~~~~                  ",
    "       ]]]]    ~~~~~~~++     ]]]]       ",
    "      ]]]]     ~~~~~~~~       ]]]]      ",
    "      ]]     ~+~~~~~~~~+        ]]      ",
    "      ]]      ~+~~~~~~~~~~      ]]      ",
    "      ]]     ~~~~~~~~~~~]       ]]      ",
    "      ]]      +~~~~~~~~~~~      ]]      ",
    "     ]]]      ~~~~~~~~~~~~      ]]]     ",
    "   ]]?       ~~~~~~~~~~   ~       ?]]   ",
    "     ]]]       ~~~~~~+~~        ]]]     ",
    "      ]]      ~~~~~~~+~~+       ]]      ",
    "      ]]      +~~~~~~~~~        ]]      ",
    "      ]]       +~+~~~~+~~+      ]]      ",
    "      ]]       ++~+~~~~~~       ]]      ",
    "      ]]]]       ~~~+         -]]]      ",
    "       ]]]]       +~         ]]]]       ",
    "                _+~~>                   ",
    "                  ~~                    ",
)
# Logo colors, as (r, g, b). The braces are the lighter green, the tree the
# darker one — change these two lines to restyle the logo.
BRACE_RGB = (129, 199, 132)
TREE_RGB = (46, 125, 50)
# Glyphs that only ever appear in the tree. A run containing any of these is
# foliage; anything else is brace. Classifying by run rather than by character
# matters because the two share characters — the art has a stray "]" inside the
# foliage and a "-" attached to a brace.
TREE_GLYPHS = frozenset("~+><_")
# A run is non-space, allowing single interior spaces; two or more spaces end it.
LOGO_RUN = re.compile(r"\S+(?: \S+)*")
ANSI_RESET = "\033[0m"

HELP_DETAILS = (
    "Commands:\n"
    "/help    Show this help message.\n"
    "/memory  Show the agent's distilled working memory.\n"
    "/session Show the path to the saved session file.\n"
    "/reset   Clear the current session history and memory.\n"
    "/exit    Exit the agent. (aliases: /quit, /bye)"
)
MAX_TOOL_OUTPUT = 4000
MAX_HISTORY = 12000
# Files the model has read are re-materialized in the prompt from disk rather
# than left to decay out of the clipped transcript. Budgets are deliberately
# larger than MAX_TOOL_OUTPUT: the transcript copy is a pointer, so these bytes
# replace the clipped ones instead of adding to them.
MAX_FILE_VIEWS = 3
MAX_FILE_VIEW_CHARS = 8000
MAX_FILE_VIEWS_TOTAL = 12000
IGNORED_PATH_NAMES = {".git", ".mini-coding-agent", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv"}


def color_enabled(stream=None):
    """True when it is safe to emit ANSI color.

    Off when output is redirected or piped, so escape codes never end up in a
    captured transcript. That also keeps the tests on the plain-text path, which
    is what lets the box-geometry assertions keep measuring real widths.
    """
    stream = sys.stdout if stream is None else stream
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    return bool(getattr(stream, "isatty", None)) and stream.isatty()


def enable_windows_ansi():
    """Turn on virtual-terminal processing so ANSI works in legacy consoles.

    Windows Terminal handles this already; older conhost does not, and without
    it the escape codes print as literal garbage. No-op everywhere else.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        # -11 = STD_OUTPUT_HANDLE, 0x4 = ENABLE_VIRTUAL_TERMINAL_PROCESSING
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x4)
    except Exception:  # noqa: BLE001 - color is cosmetic; never break startup over it
        return


def colorize_logo(line, enabled=None):
    """Color the logo's braces and foliage differently.

    Applied *after* the line has been padded and centred: ANSI codes are zero
    width on screen but count toward len(), so coloring earlier would skew every
    width calculation and bend the box out of shape.
    """
    if not (color_enabled() if enabled is None else enabled):
        return line

    def paint(match):
        run = match.group()
        red, green, blue = TREE_RGB if TREE_GLYPHS & set(run) else BRACE_RGB
        return f"\033[38;2;{red};{green};{blue}m{run}{ANSI_RESET}"

    return LOGO_RUN.sub(paint, line)


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
