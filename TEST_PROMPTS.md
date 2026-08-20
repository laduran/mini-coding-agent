&nbsp;
# Test Prompts

A running collection of prompts for manually (or programmatically) exercising `mini-coding-agent` against real models. Kept here so runs are repeatable and comparable — across models, across this project's versions, and against other tools (e.g. Open WebUI) to separate model-quality issues from harness/protocol issues.

&nbsp;
## Snake game (single-file HTML/canvas)

```text
write the famous snake game in HTML using a 2D canvas. Make sure to keep score, the game ends if the snake runs into itself or the snake hits one of the walls. Keep score. Each time the snake eats the apple, add 10 to the score. Write the program out in a single file with embedded Javascript to snake.html
```

Notes from prior runs (`qwen3.6:35b` via `mini-coding-agent`):

- Generated a game with an immediate self-collision bug: `gameLoop()` computed the "new" head as `snake[0] + (dx, dy)` before any key press, and with the initial `dx=0, dy=0` this equals the current head, triggering the self-collision check on the very first tick — game over before the player can move.
- A follow-up "the game ends immediately, please fix it" prompt exposed a separate harness bug: the model stalled re-reading the same unchanged file in an alternating-args pattern that the old duplicate-call detector didn't catch (fixed).
- The same prompt run through Open WebUI against the same model reportedly one-shot a correct result, while this project's custom `<tool>` protocol over a raw completion struggled — worth using this prompt again to compare once the harness moves closer to native tool-calling.
