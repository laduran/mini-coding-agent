&nbsp;
# Test Prompts

A running collection of prompts for manually (or programmatically) exercising `mini-coding-agent` against real models. Kept here so runs are repeatable and comparable — across models, across this project's versions, and against other tools (e.g. Open WebUI) to separate model-quality issues from harness/protocol issues.

&nbsp;
## Snake game (single-file HTML/canvas)

This is a **two-turn** scenario. Turn 2 is the interesting one — most harness bugs
found so far surfaced there, not in the initial generation. Run both, in order,
in the same session.

Suggested invocation (see the `--max-new-tokens` note below — the default used to
be too low for turn 1 to succeed at all):

```bash
uv run mini-coding-agent --cwd ~/snake-test --model qwen3.6:35b \
  --max-steps 10 --context-length 16384 --approval auto
```

**Turn 1 — generate:**

```text
write the famous snake game in HTML using a 2D canvas. Make sure to keep score, the game ends if the snake runs into itself or the snake hits one of the walls. Keep score. Each time the snake eats the apple, add 10 to the score. Write the program out in a single file with embedded Javascript to snake.html
```

**Turn 2 — fix (verbatim from session `20260808-064102-ab9d37`):**

```text
I opened snake.html in a browser. The game immediately shows "Game Over! Press Restart to play again." with a blank canvas, before I've pressed any arrow key or moved the snake at all. Please find and fix the bug.
```

Turn 2 only reproduces if turn 1 actually generated the bug, which is not
guaranteed — a 2026-08-19 run produced correct code (`initGame()` set `dx = 1`
and seeded a 3-segment snake, so the first tick collides with nothing). To force
the original condition, inject it before turn 2:

```bash
sed -i 's/dx = 1;/dx = 0;/' ~/snake-test/snake.html
```

Notes from prior runs (`qwen3.6:35b` via `mini-coding-agent`):

- Generated a game with an immediate self-collision bug: `gameLoop()` computed the "new" head as `snake[0] + (dx, dy)` before any key press, and with the initial `dx=0, dy=0` this equals the current head, triggering the self-collision check on the very first tick — game over before the player can move.
- The turn-2 prompt exposed a separate harness bug: the model stalled re-reading the same unchanged file in an alternating-args pattern that the old duplicate-call detector didn't catch (fixed, #15).
- The same turn-2 prompt then exposed the deeper cause: `history_text()` had clipped the file contents out of the prompt, so the model was re-reading to recover code it genuinely no longer had, and the loop guard blocked it (fixed, #19/#17).
- **`--max-new-tokens` must be large enough to write the whole file in one tool call.** At the old default of 512 this prompt could never succeed: the model was cut off mid-`<content>`, the closing tag never arrived, and every retry produced the same truncated output (byte-identical at `--temperature 0.2`) until the attempt budget ran out. The default is now 4096. Measured throughput for `qwen3.6:35b` Q4: ~26.7 tok/s warm, ~10.8 tok/s cold including model load; the 5470-char file cost ~1562 tokens / 58s. Measured on an RTX 4070 Laptop GPU (8GB) — the 23.9GB model does not fit in VRAM, so this is heavily offloaded and only stays usable because the model is MoE (`qwen35moe`, few active params). Treat the number as a floor, not a benchmark, and re-measure on the Arc target rather than carrying it over.
- The same prompt run through Open WebUI against the same model reportedly one-shot a correct result, while this project's custom `<tool>` protocol over a raw completion struggled — worth using this prompt again to compare once the harness moves closer to native tool-calling.
