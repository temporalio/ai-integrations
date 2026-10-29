# Execution hooks for the native agent tool design

**Status:** Draft replacement proposal for [#270](https://github.com/temporalio/ai-integrations/pull/270), starting with a design and executed proof. The proposed SDK hooks are experimental local changes, not published APIs. This draft changes no production plugin code or dependencies. Keep #270 open until its replacement is implemented and ready.

**Related issues:** [#243](https://github.com/temporalio/ai-integrations/issues/243), nested model/settings and runner defaults; [#235](https://github.com/temporalio/ai-integrations/issues/235), original-agent lifecycle listeners; [#237](https://github.com/temporalio/ai-integrations/issues/237), callback-selected handoffs.

The intended implementation keeps the clone-free provider-based runner from #270, preserving native model/settings inheritance and original agent identity, while replacing its validation interceptions with supported SDK execution hooks. The existing controls for these three behaviors pass on the local prototype. Production adoption remains blocked on agreement, implementation and release of those upstream hooks.

**Baseline:** `temporalio/ai-integrations@c6bf59e77b2db9ea2679824f927352e54e90223e`, with the locked `@openai/agents-core` 0.14.3 and Temporal TypeScript SDK 1.24.0. Node 24.22.0, pnpm 10.27.0, and Temporal CLI 1.9.1 in the existing `plugin-eval/review:e7f666534486` image. The duplicate SDK is a physical copy with a distinct `Agent` constructor.

## Decision

Keep the clone-free runner and native tool objects. Replace the three prototype interceptions and the tool-source registry with a supported upstream runtime-policy contract. The proof installs two calls in upstream's runner: one when a run consumes its effective configuration, and one when it resolves the model it will actually execute. A build check refuses SDK copies without this contract.

This is an upstream-assisted simplification. It cannot be shipped as a plugin-only change against the current 0.14.3 package. The SDK calls in this package are a disposable demonstration; editing installed dependencies is not the proposed delivery mechanism.

## Why these two points

The locked SDK's private `Runner.#resolveModelForAgent` returns an agent's `Model` object directly. A model provider only sees string model names. A before-run graph walk therefore cannot enforce the Activity boundary after a callback returns another agent or a listener changes its model.

Native `asTool()` parses parameters and calls `inputBuilder` before constructing the nested runner. Its effective configuration includes the tool's overrides, including an own `modelProvider: undefined`. The run entry sees this final merge before provider, session, or sandbox use. A shallow, frozen copy of validated sandbox configuration prevents later replacement of its client or session fields.

The model-resolution call receives upstream's selected model after its own precedence rules, including runtime agent selection and lifecycle callbacks. The Temporal policy rejects a model object and requires the Activity-backed provider before asking it to resolve a string. It checks the value that will be executed, rather than relying on a particular agent prototype or on `getSystemPrompt` being called.

One check before the run is insufficient. These two points cover different consumers.

## Proposed contract

An optional public `setRuntimePolicy` API installs a versioned policy in the execution environment. It is separate from `RunConfig`, so a nested tool's overrides cannot clear it. Every supported SDK package copy consults the same versioned registration.

The policy has two operations:

- `prepareRun(runner, agent, options)` runs before session preparation and the run loops. It checks the effective provider, run model, session and sandbox. It can return a private options snapshot. Nested native runs go through it too.
- `resolveModel(runner, agent, selectedModel)` runs before any provider or model invocation. It receives the actual executing agent and upstream's selected value, and returns the model to execute. Both ordinary and streamed paths use it. Without a policy, upstream's current behavior remains unchanged.

The registration must be scoped to the execution environment. In a Temporal Workflow it is installed by the Workflow interceptor before user modules load. The proof uses a configurable global property under a versioned registry symbol, which separate package copies can read. The first attempt used a nonconfigurable property and failed when Temporal reused a VM. The corrected form passes cold execution and replay across successive Workflows. The final upstream API still needs an explicit replacement/disposal contract for non-Temporal environments.

The bundle check examines each resolved `@openai/agents-core` package, including dependencies' separate copies, and requires its public runtime-policy entry point. Adoption must pin an upstream release with these hooks and enforce its supported version range. An older copy must fail the build; a hook on another copy cannot protect it.

## What becomes simpler

The Workflow path no longer imports `agent-tools.ts` or `check-agent.ts`. It does not replace `Agent.prototype.asTool`, `Agent.prototype.getSystemPrompt`, or `Runner.prototype.run`. Native tools retain their own object, parser registration, handlers, guardrails, approval settings and private source-agent identity.

The proof leaves the old modules and their unit tests in the disposable checkout so existing source tests still compile. They are excluded from the tested Workflow bundle. A production implementation would remove them and move the relevant assertions to the supported runtime contract. The proof does not claim the whole PR's net diff is ready for merge.

Existing cancellation propagation, the Activity-backed model provider, sandbox adapters, tracing and the clone-free runner remain. Removing the public `agentAsTool` export is a separate migration decision; this proof does not add another helper implementation.

## Developer experience and maintenance cost

Successful application code is almost identical between the current PR and this proposal. The main benefit is a smaller integration contract to maintain. The costs are an upstream dependency and later errors for some unsupported setups.

| Developer experience | Current PR | Proposed hooks |
|---|---|---|
| Agent and tool declarations | Native `agent.asTool()`, original agents, listeners and custom handoffs | Same declarations; no additional application API or internal-hook migration |
| Temporal setup | `TemporalOpenAIRunner`, Worker plugin and Activity providers | Same setup; the plugin installs the policy and bundle guard |
| Tool customizations | Original tool retains guardrails, approvals and streaming handlers | Same behavior in the executed controls |
| Unsupported nested config | Some cases fail during preflight, before the parent model runs | Checked when the nested run consumes it; the parent may already have run, and upstream returns a tool-error string |
| Multiple SDK copies | Unrecorded native tools are rejected at runtime; the demonstrated agent bypass remains | Every runtime copy must support the hooks; older copies fail bundling. Foreign native tools still fail safely because upstream does not share their inherited context |
| Dependency upgrades | Can use the PR's existing 0.14.3 dependency | Requires an upstream release and compatible dependency copies, which can delay adoption |

| Code and maintenance | Current PR | Proposed hooks |
|---|---|---|
| Integration mechanism | Three prototype replacements, native-tool source/options recording, graph traversal and repeated checks | Two SDK call sites, a public registration contract, a Temporal policy and a bundle capability check |
| Selected plugin code | 301 lines in `agent-tools.ts` and `check-agent.ts` | 46-line policy and 20-line bundle guard in the proof, plus SDK hook/registration/type changes |
| Private SDK coupling | Depends on prototype identity, system-prompt call ordering, initialization order and native-tool shape | Depends on the promised hook timing and coverage. Upstream must preserve that contract across its internal changes |
| Upfront work | Finish the current repair and its regression checks | Agree on an upstream API, implement both module formats and lifetime rules, release it, then update the plugin |
| Recurring plugin work | Recheck the private assumptions and callback ordering on each supported SDK update | Maintain the runtime policy, capability/version checks and behavioral controls against a supported SDK contract |
| Work that remains either way | Cancellation, approvals, streaming, sandbox behavior, saved-state identity and replay compatibility | Same responsibilities; hooks do not eliminate these tests or rollout costs |

The line counts describe the selected prototype mechanism, including comments and blank lines. They are not a net PR reduction, a production API size, or a measured maintenance budget. The old modules remain on disk in the disposable checkout so old tests compile, but are absent from the tested Workflow path. The expected lower recurring plugin cost is an engineering judgment based on fewer private assumptions; the work moves partly into an upstream contract and its tests. It does not disappear.

## Executed evidence

The same [Worker proof](proof.cjs), with [Workflow definitions](workflows.cjs), runs on the unchanged PR and the prototype. It records inline calls through a sink with `callDuringReplay`, model Activity names, and actual model requests. Every case is replayed with the SDK's replayer on a Worker with no Workflow cache.

| Case | Unchanged PR | Runtime-policy prototype |
|---|---|---|
| Listener changes a separate-copy agent to a model object | Inline model runs; replay calls it again; zero model Activities | Rejected; no inline calls during execution or replay |
| Handoff returns a separate-copy agent with a model object | Inline model runs; replay calls it again; only the router uses an Activity | Rejected; only the router uses an Activity |
| Same-copy listener and handoff controls | Rejected | Rejected |
| Schema or input callback overrides provider/model; explicit undefined provider | Rejected by the PR's existing checks | Rejected at the consuming runner; no inline calls |
| Separate-copy agent with a string model | One model Activity | One model Activity |
| Same-copy native nested tool | Two model Activities | Two model Activities |
| Native tool from another package copy | Rejected as unrecorded | Rejected for lacking the inherited Activity provider; no inline calls |
| Guardrails assigned to the native tool after creation | Input/output each block, guard runs once | Same behavior |
| Approval assigned after tool creation | One interruption; no nested model Activity | Same behavior |
| A loaded SDK copy lacks the new contract | Not a baseline requirement | Bundle is rejected before Workflow execution |

Results are in [baseline.txt](baseline.txt), [policy.txt](policy.txt), and the saved `histories-baseline/` and `histories-policy/` directories. The baseline's success status asserts the two defective outcomes; it does not mean the baseline is safe.

The existing focused controls cover run defaults and model inheritance, options reuse, shared-agent and nested approvals across `continueAsNew`, early and late streaming handlers, cancellation, agent listeners, custom handoffs, and sandbox lifecycle and mutation. See [controls.txt](controls.txt).

For ordinary native nested calls and a separate-copy string-model agent, the compatibility check replays both builds' histories with each build, including self-replays. It carries a waiting native nested run through Worker replacement in both directions, with no Workflow cache, and checks its completed result. A deliberately renamed Activity must fail replay with `[TMPRL1100]`. See [replay.txt](replay.txt) and [replay.cjs](replay.cjs). The rollback target here is the unchanged PR head, not the plugin's released version.

## Behavior and remaining limits

Validation of unsupported nested configurations moves from graph inspection to their actual use. For example, an explicit undefined provider now schedules the parent model Activity before returning the nested error as tool output. The unchanged PR rejected it before the parent ran. This changes error timing for unsupported setups; it does not execute the unsafe operation.

Separate SDK copies do not share upstream's inherited native-tool context. This design accounts for that by rejecting a nested run without the Temporal provider. It does not promise native nested execution across arbitrary package copies.

The prototype covers CommonJS, which is what the plugin currently bundles. An upstream implementation must add the contract to both published module formats and test the default, streaming, and sandbox model-resolution routes. It needs proper public types and upstream tests, concurrent registration/lifetime tests, and the supported dependency/runtime matrix. The proof keeps the existing same-copy `MemorySession` rejection; session implementations across package copies need their own acceptance check. Custom session support has not been redesigned here.

The main-to-#270 upgrade breaks already accepted for clone removal and native tools still apply. This proof's compatibility results compare only the current PR head with the prototype, on the named valid paths. Unsafe histories that previously executed inline, and previously rejected configurations that now reach a nested run, are not covered by that compatibility claim. This proposal does not yet close the related issues.

## Delivery

1. Propose and review the two execution hooks in `openai/openai-agents-js`. The [request draft](upstream-request.md) describes the contract; no upstream issue or PR was posted for this study.
2. After an upstream release, require that release and the bundle check in `temporalio/ai-integrations`. Replace the three prototype interceptions and registry with the policy. Keep the coupled clone-removal, identity and native-tool changes together, with their existing upgrade guidance.
3. Run the existing focused controls and the full supported CI matrix, then review the simplified implementation once. An upstream API acceptance or release is not established by this local proof.

## Rerun

Use a disposable worktree at the baseline commit, and mount it as `/work` in the named existing image. Mount this proposal worktree as `/proposal`, and preserve the image's pnpm path. On a fresh checkout, first run `make sync` in `/work/typescript/openai-agents` to restore its locked dependencies. Then run [run.sh](run.sh) inside the container with `/work/typescript/openai-agents` as its argument. It uses the project's frozen lockfile and writes only to those disposable mounts and the container's own dependency cache. No provider credentials are needed.

[prepare.py](prepare.py) contains the exact dependency and plugin changes. [runtime-policy.ts](runtime-policy.ts) is the Temporal-side policy, and [require-runtime-hooks.cjs](require-runtime-hooks.cjs) is the demonstrated bundle check. The SDK change is a prototype at two consumption points; its public export and cross-copy capability check are part of the proposal. The container image records the executed environment; it is not required to run the scripts. They need Python 3, make, Node 24.22.0, pnpm 10.27.0 and Temporal CLI 1.9.1 at `/usr/local/bin/temporal`. The dependency and source changes apply only to the disposable baseline worktree, not this proposal branch.

The same check can run without the image, using a disposable worktree at the named baseline:

```sh
git fetch origin c6bf59e77b2db9ea2679824f927352e54e90223e
git worktree add --detach /tmp/agents-runtime-proof c6bf59e77b2db9ea2679824f927352e54e90223e
cd /tmp/agents-runtime-proof/typescript/openai-agents
make sync
sh /path/to/proposal/typescript/openai-agents/design/runtime-policy/run.sh /tmp/agents-runtime-proof/typescript/openai-agents
```
