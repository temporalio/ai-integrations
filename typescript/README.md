# TypeScript integrations

Each plugin is an independent pnpm project with its own manifest and lockfile.
There is no language-level workspace or lockfile. Node.js 22.12 or later and
pnpm 10.27.0 are required.
CI selects `"22"` and `"24"` with `check-latest: true` to test the latest release
in each major series.

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

The initial histories come from sdk-typescript `main` at
`046dcef5e3d5d08710c4a07d805f17fea41e5ae5`. The separate ownership cutover renames
`@temporalio/ai-sdk` to `@temporalio/vercel-ai-sdk` and
`@temporalio/google-adk-agents` to `@temporalio/google-adk`. The other three
coordinates keep their names. Active upstream metadata is removed; development
now belongs here. Package names, subpath imports and README examples use the new
coordinates. Workflow activity names remain intact. The Google ADK random-stream
identifier follows the new package name, as explicitly requested for the cutover.

The SDK's unpublished `@temporalio/test-helpers` sources are copied at that same
commit into `tests/helpers/` in Vercel AI SDK, OpenAI Agents and Strands Agents.
Each copy builds and resolves against its plugin's environment through a private
development-only link. No plugin depends on another plugin's checkout.

Vercel's test telemetry uses its provisioned server rather than localhost:7233.
Its Web API installer supplies deterministic base64 fallbacks for the published
SDK 1.24.0; upstream added Worker-provided base64 globals after that SDK release.
Its telemetry test explicitly registers a tracing provider and an in-memory
exporter. The test environments omit the unused OpenTelemetry Node SDK and its
Jaeger/Prometheus dependencies.

The development version is `0.0.0`. The shared `release-typescript.yml` workflow
tests the tagged version across the full matrix, then publishes its tested npm
tarball through trusted publishing and verifies the public bytes and consumer
installation. Google ADK is enabled for its first Public Preview release, `0.1.0`;
the other packages remain private with final releases disabled. A new npm package
requires one interactive publication of the tested artifact before its trusted
publisher can be configured. A tag dispatch with `recover-run=<run-id>` reuses a
validated dry run or release run's tarball, so bootstrap verification and upload
retries use the original tested bytes without rebuilding. See the [TypeScript release runbook](../AGENTS.md#typescript-releases)
and the [import log](../scripts/migrate/IMPORTS.md).
