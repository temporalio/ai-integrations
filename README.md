# Temporal AI Integrations

Plugins that connect AI agent frameworks and SDKs to [Temporal](https://temporal.io) durable
execution. Each plugin is its own package with its own dependencies, tests, version and release
cadence, laid out as `<language>/<integration>/`.

| Plugin | Package | Root API | Release stage |
|---|---|---|---|
| [`go/googleadk`](go/googleadk) | [`go.temporal.io/googleadk`](go/googleadk/README.md) | `googleadk` | Public Preview |
| [`python/deepagents`](python/deepagents) | `temporalio-deepagents` | `temporalio.deepagents` | [Pre-release](https://docs.temporal.io/develop/python/integrations/deepagents) |
| [`python/google_adk`](python/google_adk) | `temporalio-google-adk` | `temporalio.google_adk` | [Pre-release](https://docs.temporal.io/develop/python/integrations/google-adk) |
| [`python/google_genai`](python/google_genai) | `temporalio-google-genai` | `temporalio.google_genai` | [Public Preview](https://docs.temporal.io/develop/python/integrations/google-genai) |
| [`python/langgraph`](python/langgraph) | `temporalio-langgraph` | `temporalio.langgraph` | [Public Preview](https://docs.temporal.io/develop/python/integrations/langgraph) |
| [`python/langsmith`](python/langsmith) | `temporalio-langsmith` | `temporalio.langsmith` | [Public Preview](https://docs.temporal.io/develop/python/integrations/langsmith) |
| [`python/mcp`](python/mcp) | [`temporalio-mcp`](https://pypi.org/project/temporalio-mcp/) | `temporalio.mcp` | [Pre-release](https://github.com/temporalio/ai-integrations/blob/main/python/mcp/README.md) |
| [`python/openai_agents`](python/openai_agents) | [`temporalio-openai-agents`](https://pypi.org/project/temporalio-openai-agents/) | `temporalio.openai_agents` | [Generally Available](https://temporal.io/blog/announcing-openai-agents-sdk-integration) |
| [`python/strands_agents`](python/strands_agents) | `temporalio-strands-agents` | `temporalio.strands_agents` | [Public Preview](https://docs.temporal.io/develop/python/integrations/strands-agents) |

More plugins are migrating here from the SDK repositories; see the target table in
[`AGENTS.md`](AGENTS.md).

Go development is documented in [`go/README.md`](go/README.md); publishing
requirements are in [`AGENTS.md`](AGENTS.md).

## Install

```
$ uv add temporalio-mcp
$ uv add temporalio-openai-agents
```

## Develop

```
$ cd python/openai_agents
$ make sync    # non-editable install into .venv (see AGENTS.md for why)
$ make lint
$ make test    # provider calls use deterministic local models and transports
```

`make help` lists every target. Conventions, CI design, release process and migration procedure
are in [`AGENTS.md`](AGENTS.md); contributor workflow is in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

[MIT](LICENSE). Each plugin directory carries an identical copy so every published package ships the license text.
