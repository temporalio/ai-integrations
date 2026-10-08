# OpenAI Codex integration for Temporal: durable Codex agent loops with host-run tools

Temporal integration for [OpenAI Codex](https://github.com/openai/codex), published as [`temporalio-openai-codex`](https://pypi.org/project/temporalio-openai-codex/) and imported as `temporalio.openai_codex`.

> **Pre-release.** It relies on Codex app-server APIs that are themselves experimental (`dynamicTools`,
> `thread/inject_items`) and on the rollout file format, so the Codex version is pinned.

Codex runs its agent loop in an external `codex app-server` process, so this integration makes the
*process* durable instead of wrapping model calls:

- **Every tool is a host tool.** Codex's own built-in tools (shell, `apply_patch`, goals, ...) are turned
  off for the thread. The tools you pass are offered to Codex as dynamic tools and run in your Workflow,
  or as Activities (`activity_as_tool`), so they are recorded, retried and, when they have side effects,
  run exactly once.
- **The conversation lives in the Workflow.** A turn is one or more *segments*. A segment is one Activity:
  it rebuilds a fresh `CODEX_HOME` from the rollout the Workflow holds, resumes the thread, injects the
  real output of the previous tool call, and runs the turn. When the model calls a tool the segment ends,
  the Workflow runs the tool, and the next segment continues. A Worker dying mid-turn just retries the
  segment from the last committed rollout; a tool that already ran is not run again.

## Install

```bash
uv add 'temporalio-openai-codex[bundled-codex]'
```

The `bundled-codex` extra ships the exact-pinned Codex binary (about 120-160 MB per platform). Without
it, set `CODEX_BIN` or pass `CodexPlugin(codex_bin=...)`.

## Use

```python
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.openai_codex import CodexPlugin
from temporalio.openai_codex.workflow import CodexSession, activity_as_tool, codex_tool


async def lookup(q: str) -> str:
    """Look up a ticket by id."""
    return f"ticket-{q}: resolved"


@activity.defn
async def refund(order_id: str) -> str:
    """Refund an order (a real side effect: runs exactly once per model call)."""
    ...


@workflow.defn
class SupportAgent:
    def __init__(self) -> None:
        self.codex = CodexSession(
            tools=[codex_tool(lookup), activity_as_tool(refund, start_to_close_timeout=timedelta(seconds=30))],
            instructions="You are a support agent.",
        )

    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await self.codex.run(prompt)).text
```

On the Worker, the plugin holds everything that must stay out of history (binary, model provider,
credentials):

```python
plugin = CodexPlugin(env={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]})
worker = Worker(client, task_queue="codex", workflows=[SupportAgent], activities=[refund], plugins=[plugin])
```

### Streaming and other frameworks

`CodexSession.run(prompt, observer_context=...)` hands an opaque JSON context to the Worker's
`observer_factory` (an async context manager yielding a `CodexObserver`), which receives
`model_interaction_started`, `reply_delta` and `model_interaction_ended` as the turn runs. Wrap tools in
your own `CodexTool(spec, handler)` to run them through another execution layer.

## Testing without credentials

`temporalio.openai_codex.testing.FakeResponsesServer` is a scripted fake of the Responses API. Point the
plugin at it with `CodexPlugin(config_overrides=fake.config_overrides)` and drive the real `codex` binary;
each `CALL:<tool>|<json>` line in the prompt is one tool call.

## Limits

- A tool call ends a segment, so each call restarts the app-server (about 0.1s) and calls run one at a time.
- `apply_patch` has no per-thread off switch; removing it needs a model-catalog override that this
  integration does not do yet, so a real model's tool list may still include it.
- Files a tool writes to disk do not follow a failover Worker; that is the tool's concern.
- The rollout is sent in full on each segment's input; very long conversations need Continue-As-New and
  payload offload.

## Documentation

- Integration guide: https://docs.temporal.io/develop/python/integrations/openai-codex
- Repository conventions: [`AGENTS.md`](https://github.com/temporalio/ai-integrations/blob/main/AGENTS.md)

## Develop

```bash
make sync   # install (non-editable) into .venv
make lint
make test   # drives the real Codex binary against a local fake Responses API; no credentials
```
