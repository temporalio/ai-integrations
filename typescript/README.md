# TypeScript integrations

Each plugin is an independent pnpm project with its own manifest and lockfile.
There is no language-level workspace or lockfile. Node.js 22.12 or later and
pnpm 10.27.0 are required; a plugin whose dependencies need more declares its
own floor in `engines` (`google-adk`: 22.17.0).
CI selects `"22"` and `"24"` with `check-latest: true` to test the latest release
in each major series; a plugin may name its floor's `major.minor` as its first
runtime instead (`google-adk` tests 22.17 and 24).

```bash
cd typescript/openai-agents
make sync
make lint
make test
make build
```

`make sync` installs the committed lockfile. `make sync-latest` selects newer
versions within the declared dependency ranges, respecting the two-week release
cooldown and the SDK's first-party and exact AI SDK family exceptions.
`make test TEST_ARGS='--match=*replay*'` selects AVA tests. Tests provision local
Temporal servers and use model doubles or local MCP servers; they need no provider
credentials. `make build` writes an npm tarball under the plugin's `dist/`.
CI checks its source, compiled exports and MIT license, then imports it from a
clean consumer without development dependencies.
Plugins can also provide `tests/smoke/index.cjs` and workflow fixtures in that
directory. The smoke runner copies them into the clean consumer, installs the
declared Temporal Worker and testing versions, and runs the entrypoint. Strands
uses this to bundle and execute a workflow with a local model against the provider
version npm selects from its peer range, independently of the development lockfile.

The SDK's unpublished `@temporalio/test-helpers` sources are copied into
`tests/helpers/` in Vercel AI SDK, OpenAI Agents and Strands Agents.
Each copy builds and resolves against its plugin's environment through a private
development-only link. No plugin depends on another plugin's checkout.

Vercel's test telemetry uses its provisioned server rather than localhost:7233.
Its Web API installer supplies deterministic base64 fallbacks for the published
SDK 1.24.0; upstream added Worker-provided base64 globals after that SDK release.
Its telemetry test explicitly registers a tracing provider and an in-memory
exporter. The test environments omit the unused OpenTelemetry Node SDK and its
Jaeger/Prometheus dependencies.

