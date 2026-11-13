# Google ADK agents — Temporal integration (Go)

Make a [Google ADK](https://google.github.io/adk-docs/) (`adk-go`) agent **durable
and replay-safe under Temporal** without rewriting the agent. The agent's
orchestration loop runs inside a Temporal Workflow; each **LLM call** becomes a
durable Temporal Activity, and any **tool** that does I/O opts into an Activity
too — so calls are retried, timed-out, visible in the Temporal UI, and replayable.

You keep building agents the native ADK way — `llmagent.New(...)` with a
`model.LLM`, `tool.Tool`s / `tool.Toolset`s and `SubAgents`, wrapped in
`runner.New(...)` and driven by `r.Run(...)`. You change two things:

1. Use `googleadk.NewModel("<model-name>")` as your agent's `Model`. It is a
   `model.LLM` whose calls are dispatched to the `InvokeModel` Activity; the real
   model is reconstructed **worker-side** (never in the workflow).
2. Pass the bridged context from `googleadk.NewContext(workflowCtx)` to `r.Run`,
   which installs Temporal-deterministic time / UUID / task-fan-out providers.

Tools run **in-workflow by default** — the idiomatic Temporal model: your workflow
is deterministic, and anything that touches the network, clock, or disk goes
through an Activity. Opt a tool into an Activity with `googleadk.ActivityAsTool`,
or use `googleadk.NewMCPToolset` for MCP. The real model and any activity/MCP tool
handlers live worker-side in the registry declared by `googleadk.Config` and
wired onto the worker by `googleadk.NewPlugin(...)`.

## Add to your project

The module path is `go.temporal.io/googleadk`, replacing
`go.temporal.io/sdk/contrib/googleadk`.

From your application's Go module, run:

```sh
go get go.temporal.io/googleadk@latest
```

```go
import "go.temporal.io/googleadk"
```

This package depends on the deterministic ADK `platform` seams
(`WithTimeProvider`, `WithUUIDProvider`, `WithTaskRunner`), `tool/toolutils.PackTool`,
and the `model.NewLLM` registry lookup from upstream `google.golang.org/adk/v2`
(the registry itself stays application-owned; this package never registers into it).
The minimum supported ADK version is v2.6.0. This package requires Go 1.26.10+
so the latest-dependency checks can also run ADK v2.7.0. The replay-safe telemetry
gate composes `workflow.IsReadOnly`.

## Samples

Runnable end-to-end samples live in
[temporalio/samples-go](https://github.com/temporalio/samples-go/tree/main/googleadk).
Those upstream samples use the former module path; update their Google ADK
import to `go.temporal.io/googleadk` when using this module.

## Hello world

Two halves: the **worker** registers the real model handler as an Activity; the
**workflow** builds a vanilla ADK agent and drives it.

```go
package main

import (
	"context"
	"log"

	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"

	"go.temporal.io/googleadk"
)

const taskQueue = "adk"

// AgentWorkflow runs a native ADK agent. The model call inside r.Run is
// dispatched to a Temporal Activity by googleadk.NewModel.
func AgentWorkflow(ctx workflow.Context, question string) (string, error) {
	// The model is a TemporalModel: in-workflow it only carries the model name;
	// the real gemini model is reconstructed worker-side by the ModelFactory.
	root, err := llmagent.New(llmagent.Config{
		Name:        "assistant",
		Description: "a helpful assistant",
		Model:       googleadk.NewModel("gemini-2.0-flash"),
		Instruction: "Answer concisely.",
	})
	if err != nil {
		return "", err
	}

	r, err := runner.New(runner.Config{
		AppName:           "hello",
		Agent:             root,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		return "", err
	}

	// NewContext bridges the workflow.Context into the context ADK reads its
	// determinism/executor seams from. Pass it straight to Run.
	adkCtx := googleadk.NewContext(ctx)
	msg := genai.NewContentFromText(question, genai.RoleUser)

	var answer string
	for ev, err := range r.Run(adkCtx, "user-1", "session-1", msg, agent.RunConfig{}) {
		if err != nil {
			return "", err
		}
		if ev != nil && ev.Content != nil {
			for _, p := range ev.Content.Parts {
				if p != nil && p.Text != "" {
					answer = p.Text
				}
			}
		}
	}
	return answer, nil
}

func main() {
	c, err := client.Dial(client.Options{})
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()

	// Worker-side registry, wired as a worker plugin: the real model lives here.
	// API keys are captured in the factory closure and never cross the Activity
	// boundary. The plugin registers the Activities at worker start and closes
	// cached MCP toolsets at worker stop.
	adkPlugin, err := googleadk.NewPlugin(googleadk.Config{
		Models: map[string]googleadk.ModelFactory{
			"gemini-2.0-flash": func(ctx context.Context, name string) (model.LLM, error) {
				// nil config reads GEMINI_API_KEY / GOOGLE_API_KEY from the env, worker-side.
				return gemini.NewModel(ctx, name, nil)
			},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	w := worker.New(c, taskQueue, worker.Options{Plugins: []worker.Plugin{adkPlugin}})
	w.RegisterWorkflow(AgentWorkflow)

	if err := w.Run(worker.InterruptCh()); err != nil {
		log.Fatal(err)
	}
}
```

`Config.Models` is optional: when a model name is absent, `InvokeModel` falls
back to ADK's model registry (`model.NewLLM`), which stays application-owned —
this package never registers into it, so your own `model.Register` calls are
honored — and `gemini-*` names the registry does not know resolve via a
built-in zero-config Gemini fallback (a `nil` config reads `GEMINI_API_KEY` /
`GOOGLE_API_KEY` worker-side). Supply a factory to inject credentials, disable
the model SDK's own retries (see below), or override the fallbacks.

The Worker resolves the model by the name passed to `NewModel`. A
`BeforeModelCallback` that changes `LLMRequest.Model` does not pick another
Worker model: the resolved model receives the new name, as ADK's models do
outside Temporal. To use another model, build the agent with
`NewModel(<name>)`.

## What you get

- **Every LLM turn is a durable Activity.** `NewModel(name)` returns a `model.LLM`
  that dispatches to the `InvokeModel` Activity. The workflow only ships the model
  **name**; the Activity reconstructs the model from your `ModelFactory`.
- **Deterministic tools, in-workflow by default.** Ordinary `functiontool.New(...)`
  tools run on Temporal's deterministic dispatcher inside the workflow — no
  Activity overhead, and their session-state mutations (`ctx.State().Set`,
  `ctx.Actions()`) propagate normally.
- **Issue workflow commands from a tool.** `WorkflowContext(ctx)` returns the
  `workflow.Context` the bridged context dispatches on, so an in-workflow tool
  can start a child workflow, set a timer, or signal another execution. Use the
  returned Context rather than closing over the workflow function's own: during
  concurrent fan-out the tool runs on a different coroutine, and blocking with
  another coroutine's Context can panic or stall workflow execution. The
  accessor reports false outside `NewContext` (a local ADK run), so a tool can
  fall back to a non-durable path.
- **Opt a tool into an Activity when it does I/O.** `ActivityAsTool(myActivity,
  ...)` exposes an existing `func(context.Context, TArgs) (TResults, error)`
  Temporal activity to the agent as a tool (parameter schema inferred from
  `TArgs`); its `Run` dispatches the activity. Register the same activity on the
  worker as usual.
- **MCP, statelessly.** `NewMCPToolset(...)` is a workflow-side proxy: it lists
  remote tools (full declarations, including parameters) via `ListMcpTools` and
  executes calls via `CallMcpTool`. The live, stateful `mcptoolset.New(...)` runs
  worker-side, never in the workflow. Your `MCPFactory` runs at most once per
  toolset name — the toolset is cached and shared across calls — and the plugin
  closes any cached toolset that implements `Close() error` at worker stop
  automatically; with manual wiring, call `Activities.Close` yourself after
  `worker.Run` returns. A tool's own error, such as an MCP `isError` result,
  reaches the model once, as it does without Temporal; `CallMcpTool` succeeds
  and is not retried. The plugin counts an error as the tool's own when it
  wraps no other error and is not an `ApplicationError`. ADK's MCP tools wrap
  the error of a failed MCP request, so that error fails `CallMcpTool` with a
  retryable `googleadk.McpError`, as does an error returned after the Worker
  stops the call. To have a failure in your own toolset or a `FakeMCPServer`
  handler retried, wrap its cause with `%w` or return a retryable
  `temporal.ApplicationError`.

  - The toolset lists its tools once per model step and preserves the command
    order and UUID consumption of Histories recorded with earlier ADK releases.
    A toolset wrapped in another that does not forward `ProcessRequest`, such
    as `tool.FilterToolset` or `tool.WithConfirmation`, lists each time ADK asks
    for its tools, in every model step. Before ADK v2.6.0, such a toolset listed
    only in the first step of an agent run. A History recorded before v2.6.0 in
    which an agent with such a toolset took a second model step in one run does
    not replay on v2.6.0 or later; one-step runs do. Use
    `MCPToolsetOptions.ToolFilter` instead of `tool.FilterToolset`, and
    `mcptoolset`'s `RequireConfirmation` option instead of `tool.WithConfirmation`.
  - From the second model step of an agent run, a toolset in the agent's
    `Toolsets` declares only the tools the server lists, and `ToolFilter`
    allows, for that step. A model call to an MCP tool it no longer declares
    gets ADK's tool-not-found error: no `CallMcpTool` Activity and no before-
    or after-tool callbacks run, but on-tool-error callbacks do. In the first
    step, such a call still runs `CallMcpTool`; for a tool the server no longer
    lists, it fails with a non-retryable `McpError`.
  - From the second model step, your own `Tools` calls on such a toolset, for
    example from an `InstructionProvider`, return the agent run's first list,
    so they don't see changes the server or `ToolFilter` makes during the run.
    To read the current list, call `Tools` on a second `NewMCPToolset` that is
    not in the agent's `Toolsets`.

  To upgrade to this version:
  - A run that used `NewMCPToolset` directly in an agent's `Toolsets` before
    you deploy this version fails replay on it with `[TMPRL1100]`, with or
    without a tool call, and so does such a run on this version after a
    rollback. Let those runs finish first, or keep them on the earlier version
    with Worker Versioning. Runs whose MCP toolsets are all wrapped replay and
    continue across the upgrade and a rollback, with the ADK release they ran
    on.
  - If an agent or a runner plugin has a tool-error callback, or an after-tool
    callback that treats errors differently from results, don't run Workers of
    an earlier version and this one on the same Task Queue. An earlier Workflow
    Worker hands this version's tool error to ADK as a result and skips the
    callback, so the run fails to replay on the other version or closes as
    `Failed`. Deploy this version on a new Task Queue or, once the runs in the
    item above have finished, stop every earlier Worker before you start this
    version's. If you roll back, keep runs that ran on this version on its
    Workers until they close.
- **Remote A2A agents.** Build ADK's remote agent (`remoteagent.NewA2A`) with
  `AgentCardProvider: googleadk.NewA2AAgentCardProvider(cardURL, activityOptions)`
  and `ClientProvider: googleadk.NewA2AClientProvider(activityOptions)`. The card
  fetch and each message then run in the `ResolveA2AAgentCard` and
  `SendA2AMessage` Activities (a cancellation in `CancelA2ATask`), so a replay
  reads them from History instead of sending them again. Like ADK's own
  provider, the card provider fetches the card on every run of the agent; set a
  static `AgentCard` to skip the fetch.
  - The Activities build a2a-go's JSON-RPC and REST clients with no HTTP timeout
    of their own, so the Activity's timeouts bound each call. To add options,
    such as call interceptors that add credentials, set
    `Config.A2AClientFactory` to `googleadk.NewA2AClientFactory(opts...)`, which
    keeps those clients. Any other `a2aclient.Factory` uses its own HTTP
    clients: a2a-go's defaults stop every call at 3 minutes and report an HTTP
    error status only as text, which the plugin retries.
  - A zero `StartToCloseTimeout` falls back to one minute, and with no
    `RetryPolicy` Temporal retries without limit. A call that fails with an HTTP
    status follows the rule under **Typed failures**. An A2A error the remote
    agent answers with is not retried; ADK turns it into an error event. A send
    that times out or fails on the network is retried, which sends the message
    again and starts another remote task, so set a `StartToCloseTimeout` longer
    than the remote agent takes to answer. The same happens when the Worker
    stops, or the Activity is paused or reset, during a send. A card file
    missing on the worker is retried too.
  - ADK streams the send unless the run's `StreamingMode` is
    `agent.StreamingModeNone`, so the zero `RunConfig` streams too. A streamed
    reply reaches the workflow when the stream ends, with the chunks of each
    artifact merged. In its default output mode, an ADK server's partial events
    are dropped, since ADK saves none of them. In `OutputArtifactPerEvent` mode
    they are kept, so a reply of thousands of chunks can pass the payload
    limit. If the stream fails after its first events, the workflow gets those
    events and the error, without a retry, and ADK cancels the unfinished task
    as it does without Temporal.
  - A cancellation gets a single 5-second attempt, and still goes out after the
    workflow is cancelled. Cancelling the workflow while a send is in flight
    does not cancel the remote task: the send's result is dropped, and ADK
    cancels only a task it has seen, so the remote agent finishes it on its
    own.
  - The client sets the message ID from the workflow's random stream when it
    sends. ADK's `BeforeRequestCallbacks` run earlier and see a2a-go's random
    ID, which differs on every replay.
- **Deterministic by construction.** `NewContext` binds ADK's `platform.Now` to
  `workflow.Now`, `platform.NewUUID` to a deterministic generator drawn from the
  workflow random stream, and `platform.RunTasks` to a `workflow.Go` fan-out, so
  the agent loop replays deterministically. Concurrent tool fan-out is on by
  default; use `NewContext(ctx, googleadk.WithSequentialToolFanout())` for a
  serial fallback.
- **Typed failures.** Model / tool / MCP / A2A failures surface as Temporal
  `ApplicationError`s tagged `googleadk.ModelError` / `.ToolError` / `.McpError`
  / `.A2AError`. An MCP tool's own error is not one of these: ADK, its
  tool-error callbacks and the model get an error with the tool's message, as
  without Temporal, not a `googleadk.McpError`. Its Go type and identity don't
  cross the Activity boundary, so `errors.Is` and `errors.As` don't match it;
  a callback can only check its message (see **MCP, statelessly** above).
  Classify with `IsNonRetryable(err)`; never string-match. Upstream HTTP status
  drives retryability (`408`/`409`/`429`/`5xx` retryable, other `4xx` not).
  The status is read from a `genai.APIError`, or else from an int `StatusCode`
  field of 400 or more on any error the model returns or wraps, as openai-go's
  errors have. A custom `model.LLM` can expose its status the same way.
  A streamed call whose chunks could not all be published fails with
  `googleadk.StreamError` (see [Streaming](#streaming)).
  A non-streamed Gemini reply with no candidates, which is how Gemini answers a
  blocked prompt, fails at once as a non-retryable `googleadk.ModelError`, like
  a permanent `4xx`: ADK returns it as `empty response`, with no status. With
  `WithStreaming`, ADK returns the block as a response with an `ErrorCode`
  instead. Other errors without a status are retried.
- **Disable model-SDK retries in your `ModelFactory`.** `InvokeModel` already runs
  under Temporal's `RetryPolicy`; leaving the model client's own retries on retries
  a transient failure twice over. Let Temporal own retries.
- **Test without a live LLM.** `testing.go` ships `FakeModel`, `FakeMCPServer`, and
  `TextResponse` / `FunctionCallResponse` so you can unit-test workflows with no
  network. The test environments do not run plugins; register the Activities
  directly with `NewActivities` + `Register` (as this repo's own tests do).

> **Determinism note.** Because plain tools run in-workflow, their code must be
> deterministic and replay-safe — no direct network, clock, randomness, or
> goroutines. Anything that isn't belongs in an `ActivityAsTool` (or an MCP tool),
> where it runs worker-side under Temporal's retry/timeout policy.

## Heartbeats

The plugin's own Activities heartbeat while they run: `InvokeModel`,
`ListMcpTools`, `CallMcpTool`, `ResolveA2AAgentCard`, `SendA2AMessage` and
`CancelA2ATask`. That's how a cancel reaches a running call, and how a lost
Worker is noticed within about 10 seconds instead of at the
`StartToCloseTimeout`.

- A zero `HeartbeatTimeout` in their Activity options defaults to 10 seconds:
  `WithModelActivityOptions`, `MCPToolsetOptions.ActivityOptions`,
  `NewA2AAgentCardProvider` and `NewA2AClientProvider`. A timeout you set wins.
- A negative `HeartbeatTimeout` turns heartbeats off. A cancel then doesn't
  reach a running call.
- They heartbeat on a timer, a third of the timeout apart, not on progress. So
  a slow model or MCP server isn't mistaken for a lost Worker, and a stuck one
  still runs until the `StartToCloseTimeout`.
- In Temporal Cloud, each heartbeat that reaches the server is an
  [Action](https://docs.temporal.io/cloud/actions). The first one comes a third
  of the timeout into a call, so a call shorter than that, about 3 seconds by
  default, sends none. After that, the SDK holds heartbeats back to one every
  0.8 times the timeout (8 seconds by default), or every
  `MaxHeartbeatThrottleInterval` of the Worker if that's shorter. So with the
  default Worker options, a 30-second call that succeeds sends four. A call
  that fails or is canceled also sends the one the SDK was holding back as it
  ends: a 30-second call that fails sends five, the last two under 3 seconds
  apart. Each retry is a new attempt, and heartbeats again.
- `CancelA2ATask` gets the same default, but the server caps its heartbeat
  timeout at the cancellation's 5-second `StartToCloseTimeout`. Nothing cancels
  a cancellation, so its heartbeats only notice a lost Worker, and only a
  cancellation slower than about 1.7 seconds sends one.
- `ActivityAsTool` runs your own Activity with the options you give it, so it
  gets no default.

## Human-in-the-loop (HITL) tool confirmation

A tool that needs approval calls ADK's `ctx.RequestConfirmation(hint, payload)`.
ADK records the request and emits a function call named `adk_request_confirmation`,
ending the turn. Because the tool runs in-workflow, the request lands in the
workflow's own event actions. MCP tools participate too: a confirmation the
worker-side tool requests (e.g. via `mcptoolset`'s `RequireConfirmation` option)
is tunneled back across the Activity boundary and re-recorded workflow-side, so
the agent pauses the same way. Drive it from your workflow like this:

```go
for {
	var events []*session.Event
	for ev, err := range r.Run(adkCtx, userID, sessionID, msg, agent.RunConfig{}) {
		if err != nil {
			return err
		}
		events = append(events, ev)
	}

	pending := googleadk.PendingConfirmations(events)
	if len(pending) == 0 {
		break // done
	}

	// Ask the human. Deliver decisions via a Temporal signal or update.
	var decisions []googleadk.ConfirmationDecision
	for _, p := range pending {
		var d googleadk.ConfirmationDecision // {FunctionCallID, Confirmed}
		// e.g. workflow.GetSignalChannel(ctx, googleadk.ConfirmationSignalName).Receive(ctx, &d)
		decisions = append(decisions, d)
	}

	// Resume: re-run with the confirmation responses.
	msg = googleadk.ConfirmationResponse(decisions...)
}
```

`PendingConfirmations` exposes each pending call's `OriginalCall` and `Hint` for
your UI; `ConfirmationResponse` builds the resume message ADK expects.

## Continue-as-new (long conversations)

A conversation's history lives in the ADK session. To keep a workflow's history
bounded, snapshot the session and continue-as-new:

```go
if workflow.GetInfo(ctx).GetContinueAsNewSuggested() {
	snap, err := googleadk.ExportSession(adkCtx, svc, appName, userID, sessionID)
	if err != nil {
		return err
	}
	return workflow.NewContinueAsNewError(ctx, AgentWorkflow, snap /* + next input */)
}
```

On the next run, rebuild the session before driving the agent:

```go
svc := session.InMemoryService()
if snap != nil {
	if _, err := googleadk.ImportSession(adkCtx, svc, snap); err != nil {
		return err
	}
}
```

`SessionSnapshot` is JSON-serializable (session-scoped state + full event
history); every value in session state and every tool result must be
JSON-encodable. App/user-scoped state (managed across sessions by the session
service) is not carried — use a durable session service for that.

## Streaming

`NewModel(name, googleadk.WithStreaming(topic, 0))` drives the model in streaming
mode: the `InvokeModel` Activity calls the model with `stream=true`, heartbeats,
and **publishes each chunk** to a per-run
[`workflowstreams`](https://pkg.go.dev/go.temporal.io/sdk/contrib/workflowstreams)
topic for external (UI) consumers, then returns the aggregated final response into
the workflow so replay stays deterministic.

Call `googleadk.StreamServer(ctx)` once near the top of the workflow that drives
`r.Run`, and set `agent.RunConfig{StreamingMode: agent.StreamingModeSSE}`:

```go
func StreamingAgentWorkflow(ctx workflow.Context, q string) (string, error) {
	if err := googleadk.StreamServer(ctx); err != nil { // required when streaming
		return "", err
	}
	topic := "run-" + workflow.GetInfo(ctx).WorkflowExecution.ID
	root, _ := llmagent.New(llmagent.Config{
		Model: googleadk.NewModel("gemini-2.0-flash", googleadk.WithStreaming(topic, 0)),
		// ...
	})
	// ... build runner, set agent.RunConfig{StreamingMode: agent.StreamingModeSSE}, drive r.Run as above
}
```

External consumers read chunks with `workflowstreams.NewClient(c, workflowID, ...).Subscribe(...)`.
The bidirectional `RunLive` path (hard-coded goroutines/channels) is **not** supported.

If a batch of chunks cannot be published, the model call fails with
`googleadk.StreamError` instead of returning as if every chunk was delivered. It
is non-retryable when a retry would call the model again and fail the same way:

- **A batch over the server's size limits**: 2 MiB per payload and 4 MiB per
  gRPC message by default. Configure external storage on the Worker's Client
  and on every subscriber's Client so large batches are stored, or keep
  responses smaller.
- **A run past its Signal limit.** Each batch is one Signal, and a run accepts
  10,000 Signals by default, counting the Workflow's other Signals. A
  long-running streaming Workflow, such as a chat session, should
  continue-as-new well before that, or use a longer batch interval
  (`WithStreaming(topic, interval)`; the default is 2 seconds).
- **A batch Workflow Streams gave up on** after its fixed 10-minute retry limit.
  This needs a model Activity `StartToCloseTimeout` over 10 minutes (the default
  is 2 minutes), and then a server outage that long ends here too.

Other failures, such as a shorter server outage, are retried; the retry calls
the model again and publishes its stream again.

## Composing with other plugins

This integration's plugin only registers its Activities at worker start and
closes cached MCP toolsets at worker stop — no interceptors, no data converter —
so it composes with other entries in `worker.Options.Plugins` (e.g. interceptor-
or converter-based plugins like `sdk-go/contrib/opentelemetry`) without conflict.
On the ADK side, add other ADK plugins to `runner.PluginConfig.Plugins` as usual.
ADK also emits its own OpenTelemetry telemetry from inside the workflow — see the
next section before wiring exporters.

## Telemetry and replay

ADK records its telemetry through the **OpenTelemetry process globals** from code
that runs **inside the workflow**: at this adk-go pin that means spans (with
token usage as `gen_ai.usage.*` span attributes and span durations as latency)
and `gen_ai.*` log events; OTel **metrics** are pending upstream
([adk-go#479](https://github.com/google/adk-go/issues/479)). Workflow code
re-executes on every history **replay** — worker restarts, redeploys, sticky-cache
eviction — and each replay re-reads the recorded Activity results from history and
re-emits identical telemetry, so observed counts inflate by one full copy per
replay, without bound over a workflow's lifetime. Temporal's replay-safe telemetry
(`workflow.GetMetricsHandler`, replay-gated tracing interceptors) never applies,
because ADK bypasses it via the OTel globals it captured at package init.

Wrap your real providers in this package's replay-safe wrappers and install them
as the globals. The log and metric wrappers drop any emission whose context
carries a replaying workflow (recovered from the context bridged by
`googleadk.NewContext`); the tracer wrapper re-creates workflow spans during
replay and suppresses their `End` instead (the span contract below). Everything
else — worker, client, and Activity telemetry — delegates unchanged:

```go
import (
	"context"

	"go.opentelemetry.io/otel"
	otellogglobal "go.opentelemetry.io/otel/log/global"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"

	"go.temporal.io/googleadk"
)

func main() {
	// NewReplaySafeTracerProvider builds and owns the tracer provider, so it
	// always installs the span-ID generator that gives a workflow span
	// re-created on replay the same trace and span IDs it drew on first
	// execution (span contract below). Pass the usual sdktrace options
	// (exporters, resource, sampler). The logger and meter wrappers wrap your
	// own providers.
	tracerProvider := googleadk.NewReplaySafeTracerProvider( /* sdktrace.WithBatcher(exporter), ... */ )
	myLoggerProvider := sdklog.NewLoggerProvider( /* processors ... */ )
	myMeterProvider := sdkmetric.NewMeterProvider( /* readers ... */ )

	// Must be the FIRST global providers set in the process: ADK captures the
	// global proxy tracer/logger at package init, and the proxy binds its
	// delegate on the first Set call only — a provider installed earlier
	// permanently bypasses the wrappers.
	otel.SetTracerProvider(tracerProvider)
	otellogglobal.SetLoggerProvider(googleadk.NewReplaySafeLoggerProvider(myLoggerProvider))
	otel.SetMeterProvider(googleadk.NewReplaySafeMeterProvider(myMeterProvider))

	// You own the tracer provider; shut it down after your clients and workers
	// stop so buffered spans flush.
	defer func() { _ = tracerProvider.Shutdown(context.Background()) }()

	// ... Temporal client, worker, googleadk.NewPlugin wiring as usual ...
}
```

`googleadk.NewPlugin` logs a best-effort warning at worker start and at
workflow replayer creation when a global provider is a raw OTel SDK provider
installed unwrapped.
Only that positively-recognized case warns; wrapped, no-op, never-set, and
custom providers stay silent. The check is best-effort — the OTel global proxy
binds its delegate on the first `Set*Provider` call permanently, so it cannot
see through a process that sets a global more than once — but a raw SDK
provider installed once and never wrapped, the realistic misconfiguration, is
recognized.

With the wrappers installed, replays add nothing. Point telemetry (log events,
metric recordings) is recorded on first execution, **at-least-once** rather
than exactly-once: a workflow task that fails or times out after emitting
re-executes live and records again — the same semantics and caveat as
`workflow.GetMetricsHandler`.

Spans get the same at-least-once contract through a different mechanism,
because ADK holds each span open across the model call's Activity await. A
span started from sequenced workflow code is a real span in every execution
mode — a replay re-creates it with a workflow-time start and the same trace and
span IDs it drew on first execution (the owned provider always installs the
span-ID generator) — but its `End` is suppressed while the workflow is
replaying. Tracers derived from a workflow span
(`span.TracerProvider().Tracer(...)`) produce spans under the same contract.
Each span is therefore exported by whichever execution reaches its `End` live:

- **Sticky-cache eviction (graceful):** eviction teardown exports nothing —
  the SDK marks teardown as replay before coroutine defers run, so ADK's
  deferred force-End of a still-open span is suppressed. The catch-up replay
  on the next workflow task re-creates the span and its live `End` exports it
  exactly once, complete: a `generate_content` span evicted mid-model-call
  keeps its `gen_ai.usage.*` token attributes, its parent linkage, and its
  original identity. With `worker.SetStickyWorkflowCacheSize(0)` (or
  `WORKFLOW_CACHE_SIZE=0` in this repo's tests) every model call straddles an
  eviction and span counts, attributes, and point telemetry all stay exact.
- **Worker shutdown, crash, or redeploy:** a span open at that moment is
  recovered the same way — the catch-up replay on the next worker re-creates
  it and exports it once when its `End` runs live. Spans are lost only when
  their workflow never continues (terminated or timed-out runs).
- **Workflow task retry:** a task that fails after a span ended re-executes
  live and exports the span again — at-least-once, the caveat all point
  telemetry shares — but the copies carry one span ID, so ID-deduplicating
  trace backends collapse them; span-count pipelines (e.g. the spanmetrics
  connector) see one extra copy per retried task, the same as they do for
  retried metric and log recordings.

`NewReplaySafeMeterProvider` is forward-looking: it gates synchronous
instrument recordings and reports their `Enabled` false while suppressed
(observable instruments pass through — their callbacks never run under a
workflow context), covering both your own workflow-side recordings through
the global meter today and ADK's metrics once adk-go#479 lands.

Telemetry from **query handlers** and **update validators** always records:
the gate composes `workflow.IsReplaying` with `!workflow.IsReadOnly`
(Experimental). Both are once-per-request operations that never re-execute
from history — a query served right after a catch-up replay (e.g. an agent
awaiting `InvokeModel` on a restarted worker) still observes `IsReplaying`
true, because the flag retains whatever the last processed history event left
there, but `IsReadOnly` excludes read-only contexts from suppression, so the
recording is kept rather than lost. Side-effect functions are read-only
contexts too, harmlessly so: they never execute during replay at all — their
recorded markers supply the value.

**OTel Logs API status:** `NewReplaySafeLoggerProvider` is built on the
pre-1.0 `go.opentelemetry.io/otel/log` (`v0.19.x` at this pin), which may
change shape between minor releases; upgrading it can require a matching
upgrade of this package. A surface test in this package fails on any method
an upgrade would newly pass through ungated.

## Supported & not-yet-supported

- **Supported:** single- and multi-agent (`SubAgents`) trees, in-workflow function
  tools, `ActivityAsTool`, stateless MCP, remote A2A agents built with the
  plugin's providers, Gemini built-in tools (executed server-side inside
  `InvokeModel`), HITL tool confirmation, continue-as-new state carry, the
  in-memory session service, and SSE streaming.
- **Not yet:** `RunLive` (bidirectional streaming), sub-agent-as-child-workflow,
  live memory/artifact tools that require in-workflow network I/O, and DB/Vertex
  session services. These raise or are documented rather than silently degrading.
- **Rejected:** ADK's remote A2A agents built without
  `NewA2AClientProvider`, or that fetch their card without
  `NewA2AAgentCardProvider` (a static `AgentCard` is fine). This includes every
  agent from the deprecated `agent/remoteagent` package. They fetch their agent
  card and send messages from Workflow code, so every replay would send them
  again. A model call from an agent with such a remote agent below it returns an
  error and schedules no Activity. Below means among its sub-agents, at any
  depth, or wrapped with `agenttool` by one of their tools or the call's own.
  That check misses a remote agent that is not below an agent making a model
  call: a remote root agent, one under a sequential, loop or parallel agent,
  and one reached by transfer. Model callbacks can also get past it: a
  `BeforeModelCallbacks` entry that answers instead of the model skips the
  check, and an `OnModelErrorCallbacks` entry can turn the refusal into a
  transfer to the remote agent. Only `googleadk.NewRunnerPlugin(root)` catches
  these: add it to the runner's plugins, and it fails every run, before any
  agent runs, while the tree under `root` holds such an agent. It sees an
  agent wrapped with `agenttool` in an agent's `Tools`, but not in its
  `Toolsets`. Neither check sees a remote agent further inside an agent that
  `agenttool` wraps, such as one under a sequential, loop or parallel agent
  there: ADK gives no way to reach the wrapped agent's own sub-agents. An agent
  on `NewModel` inside it still checks the agents below it at its own model
  calls. For the requests that still go out, `NewContext` logs one warning,
  outside replay, the first time an HTTP request carries its context. It only
  warns: the request still goes out.

  An open run that made such a model call on an earlier version fails to replay
  on this one with a non-determinism error. Let those runs finish before you
  upgrade, or keep them on the earlier version with Worker Versioning.
