# DeepSeek Harness (dsh) — Anatomy Note

Studied at `0.1.1-rc.2`, commit `b150a551b`, `~/source/deepseek-harness`.
Read-only pass: nothing built or executed.

Cordis-based: every capability is a plugin row in a `cordis.yml`. `packages/core/agent-loop`
is explicitly *"the only package in the harness that contains concrete loop logic"*
(`packages/core/agent-loop/README.md`). Everything else is an abstract service or a plugin
against an extension point.

---

## 1. Agent loop

**Design choice: three nested levels, and no step ceiling — the model decides when to stop.**

| level | code | what it drives |
|---|---|---|
| `kick()` | `agent.ts:212` — `while (await this.turn()) {}` | turns, until the inbox is empty |
| `turn()` | `agent.ts:263` — `while (true)` | steps within one turn |
| `step()` | `agent.ts:339` — `while (true)` | **retry only**, one model call per pass |

One full iteration:

1. `preStep()` (`agent.ts:225`) claims queued messages, assembles the system prompt via
   `systemPrompt.assemble()` (`:230`), and fires an `agent/pre-step` waterfall a plugin can
   reject.
2. `buildRequest()` (`:426`) freezes one request from the declared route plus
   `session.deriveMessages()`.
3. `llm.stream(request)` (`:346`); each chunk is appended to the session as `assistant/chunk`
   (`:350`) and fed to a `BlockAssembler`.
4. Tool calls are read as **typed blocks**: `message.content.filter(block => block.type === 'tool-call')` (`:412`).
5. `executeToolCalls()` (`:414`) runs them; results re-enter as context.
6. Return to 1 unless the step concluded.

**Termination** — three exits, none of them a counter:

- Zero tool-call blocks → `{ kind: 'completed' }` (`:413`)
- Provider hit its ceiling → `{ kind: 'max-tokens' }` (`:410`), *sticky* so a later clean step
  cannot downgrade the turn outcome (`:285-290`)
- A tool sets `concludesTurn` (`tool-calls.ts:157`) — a tool can end the loop

`grep -rniE "maxstep|stepLimit|maxIterations"` over `agent-loop` and `agent` returns **nothing**.
There is no runaway backstop. The loop trusts the model to stop emitting tool calls.

Cancellation is an `AbortSignal` checked at ~8 points via `signal.throwIfAborted()`, with a
**fresh `AbortController` per turn** (`:325`) so a latch set on the old one goes stale.

---

## 2. Tool protocol

**Design choice: native provider tool-calling, with deliberately permissive argument handling.**

No text-format parsing — no `<tool>` tags, no regex. Tool calls arrive as structured content
blocks from the provider and are filtered by type (`agent.ts:412`).

Argument parsing is the surprise (`tool-calls.ts:104`):

```ts
function parseArguments(raw: string): unknown {
  try { return raw ? JSON.parse(raw) : {} }
  catch { return raw }        // invalid JSON preserved as text, not rejected
}
```

Malformed JSON is **passed through as raw text** rather than raising. The decision moves to the
tool, which can produce a meaningful error, instead of the loop rejecting generically.

Execution (`tool-calls.ts:199`) is a bounded parallel pool sized by `maxParallelToolCalls`, with
an *exclusive barrier* mode for tools that must run alone. Results and contexts **commit in model
order** regardless of completion order (`:113-115`). Calls skipped by an abort are recorded
explicitly via `appendSkippedToolCall` (`:96`, `:240`) rather than silently dropped.

There is also a **code mode** (`packages/core/tools/src/code-mode.ts`): a single `run_code` tool
whose schema is generated per language, so the model writes code against an SDK instead of
selecting among many tool schemas.

---

## 3. Context management

**Design choice: event-sourced session log; the message list is a derived view, not the store.**

Every state change is an appended event — `turn/start`, `step/start`, `user/message`,
`assistant/chunk`, `assistant/message`, `step/end`, `turn/end` — carrying `seq`, `surfaceOp`, and
`sourceEventSeqs` linking a message back to the chunks that produced it (`agent.ts:400-409`).
The prompt's message list comes from `session.deriveMessages()` (`:341`).

Three separate mechanisms keep the window bounded, all pluggable seams rather than loop logic:

- **`compaction/`** — `compaction-basic` does token-pressure summarization; `compaction-tool-result-pruner`
  is a *model-free* pruner; `command-compact` is the human trigger.
- **`spill/`** — oversized tool output is persisted out of line and the inline result replaced with
  *"a bounded preview and retrieval locator."*
- **`context/`** — plugins that add model-visible context without defining a tool
  (`agent-instructions`, `time-context`, `file-reference`, `session-reference`).

Cross-session: `session-reference` exposes bounded snapshots of *other* sessions, and session
persistence backs `resume`, which reconstructs history and continues turn numbering.

---

## 4. Model abstraction

**Design choice: twin adapters behind one seam; a self-hosted endpoint is configuration, not code.**

`ctx.llm` is the seam; adapters register provider *routes* on it
(`packages/llm/README.md`):

- `llm-deepseek` — direct first-party adapter
- `llm-pi-ai` — generic multi-provider adapter over `@earendil-works/pi-ai`

The `llm-pi-ai` README states the design goal outright: a route pi-ai does not ship
*"is declared outright, so an OpenAI-compatible gateway, a self-hosted server, or a provider newer
than the installed catalog is configuration rather than a code change."*

```yaml
- id: llm
  name: '@deepseek-ai/dsh-llm-pi-ai'
  config:
    providers:
      openai:
        apiKeyEnv: OPENAI_API_KEY      # a credential *reference*, not a secret
        baseURL: https://proxy.example.com:8443
        retryPolicy: { mode: normal, maxRetries: 3 }
```

Omitting `apiKeyEnv` leaves the route unauthenticated — which is exactly what a local Ollama or
vLLM needs. `discovery.ts` interrogates an endpoint's model listing, shaped so *"a gateway, a
self-hosted server, and the official endpoints all agree"* (`discovery.ts:17`).

**Local models are supported by design**, not incidentally.

---

## 5. Failure handling

**Design choice: every failure is structured, and retry is a plugin decision rather than loop logic.**

| failure | handling |
|---|---|
| Malformed tool arguments | preserved as raw text, passed to the tool (`tool-calls.ts:107`) |
| Provider request error | `agent/request-error` waterfall → a plugin may return `{kind:'retry'}` → `continue` (`agent.ts:373-389`) |
| Non-LLM exception | flattened to `errorChain(error)` under code `UNKNOWN` (`agent.ts:307-314`) |
| Abort mid-stream | partial content persisted with `interrupted: true` (`agent.ts:356-368`) |
| Abort mid-tool-batch | drains started calls, commits their contexts, records skipped ones (`tool-calls.ts:113-118`) |
| Turn-level error | sets `turnEnds = {kind:'error'}`, rethrows; `kick()` contains it at the driver boundary (`:213-215`) |
| Runaway loop | **not handled** — no step ceiling, no duplicate-call detection |

The retry point is the notable one: the loop does not decide. It raises `agent/request-error` and
a plugin (`llm-retry`, with per-route `retryPolicy`) decides. Absent that plugin, the error throws.

---

## Tradeoffs & takeaways

Compared against `mini-coding-agent` (the harness I work on) and Claude Code's source:

**1. No loop guard, because it assumes a strong model.** dsh has neither a step ceiling nor
duplicate-call detection. `mini-coding-agent` needs both — issues #15/#17 exist because a local
model burned 10 steps re-reading one file. dsh is built for a frontier model that stops on its own.
*Takeaway: a loop guard is a local-model tax, not a universal harness feature. Keep ours, but know
why it's there.*

**2. Native tool-calling removes an entire failure class.** dsh reads typed `tool-call` blocks;
`mini-coding-agent` parses `<tool>…</tool>` out of raw completion text. Every bug in our
`TEST_PROMPTS.md` notes — malformed JSON, truncated tags, the whole of #21 — is a consequence of
that choice. *Takeaway: the strongest fix for #21 is not better truncation detection, it's moving
to native tool-calling.*

**3. Permissive argument parsing is the opposite of ours.** dsh preserves invalid JSON as text and
lets the tool respond; our `validate_tool()` rejects up front. Ours gives a cleaner error, theirs
gives the model something to react to. Neither is obviously right — worth testing which recovers
better on a weak model.

**4. `spill/` is a better-factored version of what we built in #19.** Persist oversized output,
replace inline with preview plus retrieval locator. That is the same shape as our materialized file
views, but generalized to *all* tool output and separated into its own capability. If our file
views ever grow, this is the design to copy.

**5. Event-sourced log vs. list-of-dicts.** Our session is a list rewritten to JSON on every
append; dsh's is an append-only event log with `seq` and provenance links, from which messages are
*derived*. That is what makes replay, compaction, and partial-output recovery tractable. It is also
substantially more machinery than a pedagogical harness needs.

**6. Retry as an extension point.** Ours is hardcoded in `ask()`. Theirs is an event a plugin
answers. The plugin architecture is the whole thesis of this codebase — worth noting how far it is
carried, and that it costs a great deal of indirection to read.

**Trust note:** the shipped `cordis` preset states that `cordis_mount` evaluates model-written
JavaScript against the live runtime, and to *"treat a session on this preset as shell access"*
(`apps/cli/config/agent-presets/cordis/agent.cordis.yml`). Their words. Relevant before running it.
