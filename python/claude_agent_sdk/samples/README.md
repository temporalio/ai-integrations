# Samples

A chat with Claude that runs Claude Code in a [Docker Sandbox](https://docs.docker.com/ai/sandboxes/)
and asks you before each Bash call. It also gives Claude a custom `roll_dice` tool that runs an
Activity, and asks you before each roll too. It talks to the real Anthropic API, so it is not part of
`make test`; the tests in `../tests` drive the same code against a scripted API.

- `workflow.py`: `AgentWorkflow`, a `ClaudeSDKClient` conversation in a Workflow.
- `worker.py`: the Worker, which creates one sandbox per conversation.
- `run.py`: starts a conversation and sends it each line you type.
- `docker_sandbox.py`: `DockerSandbox`, a `Sandbox` backed by `sbx`.

## Run

Needs `sbx`, signed in, and the Temporal CLI. Store your Anthropic API key in `sbx` once, so it
stays out of your shell history and out of the sandboxes:

```sh
sbx secret set -g anthropic
```

Then, from this plugin's directory, run each of these in its own terminal:

```sh
make sync && temporal server start-dev
UV_NO_EDITABLE=1 uv run --locked python -m samples.worker
UV_NO_EDITABLE=1 uv run --locked python -m samples.run
```

Instead of the `sbx` secret, you can set `ANTHROPIC_API_KEY`, or `CLAUDE_CODE_OAUTH_TOKEN` from
`claude setup-token`, for the Worker, which writes it into each sandbox.

Set `CLAUDE_SANDBOX_TEMPLATE` to start sandboxes from your own `sbx` template.
