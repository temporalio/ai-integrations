# Temporal AI Integrations

Plugins that connect AI agent frameworks and SDKs to [Temporal](https://temporal.io) durable
execution. Each plugin is its own package with its own dependencies, tests, version and release
cadence, laid out as `<language>/<integration>/`.

| Language | Plugin | Package | Root API | Maturity | Docs |
|---|---|---|---|---|---|
| Python | [`python/openai_agents`](python/openai_agents) | [`temporalio-openai-agents`](https://pypi.org/project/temporalio-openai-agents/) | `temporalio.contrib.openai_agents` | GA | [OpenAI Agents SDK](https://docs.temporal.io/develop/python/integrations/openai-agents) |

More plugins are migrating here from the SDK repositories; see the target table in
[`AGENTS.md`](AGENTS.md).

## Install

```bash
uv add temporalio-openai-agents
```

Until the Temporal Python SDK release that stops bundling `temporalio.contrib.openai_agents`,
do not install this package next to `temporalio<=1.32`: both ship the same files, and
uninstalling the plugin then deletes files the SDK still needs (repair with a reinstall of
`temporalio`). Pre-releases are published to TestPyPI only for that reason.

## Develop

```bash
cd python/openai_agents
make sync    # non-editable install into .venv (see AGENTS.md for why)
make lint
make test    # offline: HTTP traffic is replayed from committed cassettes
```

`make help` lists every target. Conventions, CI design, release process and migration procedure
are in [`AGENTS.md`](AGENTS.md); contributor workflow is in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

[MIT](LICENSE). Each plugin directory links to this file.
