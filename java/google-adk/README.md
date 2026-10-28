# Temporal Google ADK

Public Preview integration between [Google ADK Java](https://github.com/google/adk-java) and
Temporal. Agents built with ADK's own `LlmAgent`, `Runner`, tools, and toolsets run inside a
Temporal workflow: every model call and every MCP call executes as a Temporal Activity, the
runner's ids and timestamps come from the workflow, and ADK's parallel fan-out runs on workflow
threads, so a run is durable, retried on failure, and replayed deterministically.

APIs and configuration may change before General Availability.

The coordinate is `io.temporal:google-adk` on Maven Central. CI tests Java 17 and 25; published
bytecode targets Java 17.

## Requirements and installation

- Java 17 or later.
- Google ADK Java 1.12.0 or later: this is the first release whose `Runner` accepts an
  `InstantSource`, a `UuidProvider`, and an RxJava `Scheduler`, which the integration relies on.
- Temporal Java SDK 1.40.0 or later. The integration compiles against the SDK but does not pull
  it in; your application supplies it.

```groovy
implementation 'io.temporal:google-adk:0.1.0'
implementation 'io.temporal:temporal-sdk:1.40.0'
```

## Hello world

Workflow code builds a vanilla ADK agent whose model is a `TemporalModel`, and drives ADK's
`Runner` through `TemporalAdk`:

```java
public class ChatWorkflowImpl implements ChatWorkflow {
  @Override
  public String chat(String message) {
    LlmAgent agent =
        LlmAgent.builder()
            .name("assistant")
            .model(TemporalModel.create("gemini-2.5-flash"))
            .instruction("Answer concisely.")
            .build();
    Runner runner = TemporalAdk.runnerBuilder().appName("chat").agent(agent).build();
    List<Event> events =
        TemporalAdk.run(runner, "user-1", "session-1", Content.fromParts(Part.fromText(message)));
    return events.get(events.size() - 1).content().map(Content::text).orElse("");
  }
}
```

The worker and the client share one plugin. Register it on the client; workers created from that
client inherit it. The plugin registers the model and MCP activities on every worker and extends
the client's data converter so ADK's request, response, event, and session types travel as JSON.
Models are created on the worker, so credentials never enter workflow history:

```java
GoogleAdkPlugin plugin =
    GoogleAdkPlugin.newBuilder()
        .model("gemini-2.5-flash", () -> Gemini.builder().modelName("gemini-2.5-flash").build())
        .build();
WorkflowClient client =
    WorkflowClient.newInstance(
        service, WorkflowClientOptions.newBuilder().setPlugins(plugin).build());
WorkerFactory factory = WorkerFactory.newInstance(client);
Worker worker = factory.newWorker("adk-task-queue");
worker.registerWorkflowImplementationTypes(ChatWorkflowImpl.class);
factory.start();
```

A model name without an explicit registration is resolved by ADK's `LlmRegistry`, which builds
Gemini models from the environment (`GOOGLE_API_KEY`, or the Vertex AI variables). Register a
factory with `modelFactory(...)` to change that. Disable the model SDK's own retries in your
factory; Temporal's retry policy on the activity replaces them.

## What the integration gives you

### Model calls as activities

`TemporalModel` is an ADK `BaseLlm`. Each `generateContent` call runs the
`googleadk.InvokeModel` activity with the model name, the `LlmRequest`, and the streaming flag;
the worker resolves the model and returns every response it produced. The default activity
options are a two-minute start-to-close timeout; set your own, including a retry policy and a
heartbeat timeout (the activity heartbeats while the model runs), with
`TemporalModel.newBuilder(name).activityOptions(...)`. The activity summary shown in the Temporal
UI is `InvokeModel: <model>` unless `summary(...)` computes one from the request.

The request's tool instances stay in the workflow; they are not part of the activity's payload. On
the worker the integration rebuilds `LlmRequest.tools()` from the function declarations in the
request's config as declaration-only stand-ins, so models that read the tool map (ADK's Claude,
LangChain4j, and Spring AI adapters) see the same tools as Gemini, which reads the config. If a
Spring AI chat model executes tools itself (a tool advisor, or an older Spring AI with internal tool
execution), those stand-ins cannot run on the worker; keep tool execution with ADK, which is Spring
AI 2.0's default.

Streamed calls (`RunConfig.StreamingMode.SSE`) use the same activity: the worker collects the
partial chunks and the final response, and the workflow replays them in order, so ADK's partial
events keep their shape. Chunks reach the workflow only when the activity completes; the
integration does not stream tokens out of a workflow. Live (bidirectional) connections are not
supported.

### Running the agent

`TemporalAdk.runnerBuilder()` returns ADK's `Runner.Builder` with the workflow's clock
(`Workflow.currentTimeMillis()`), an id source seeded once from `Workflow.newRandom()` (so the
number of ids ADK mints never shifts the ids Temporal assigns to activities and timers), a
workflow-thread scheduler, and an in-workflow `InMemorySessionService` installed. Everything else about the runner, including
`app(...)` for resumability or compaction, is ADK's own API. Do not set a scheduler on a
`ParallelAgent` or an executor on an `LlmAgent` inside a workflow: agent-level settings override
the runner's scheduler and would move ADK's fan-out onto ordinary threads.

`TemporalAdk.run(...)` subscribes to `runner.runAsync(...)` on the workflow thread and waits with
`Workflow.await`, returning the events of the turn; `TemporalAdk.await(flowable)` does the same for
any ADK flow. Never call `blockingGet()` or similar on a flow that reaches a model, a tool, or an
activity inside a workflow: that blocks a workflow thread on a lock and trips the deadlock
detector. The convenience overload creates the
session on first use (`autoCreateSession`); pass your own `RunConfig` to change that.

ADK runs `ParallelAgent` sub-agents and `PARALLEL_SUBSCRIBE` tool calls on the invocation's
scheduler. `WorkflowScheduler.parallel()` (the default) starts each task on its own workflow
thread, so sibling model calls are scheduled together and wait concurrently;
`TemporalAdk.runnerBuilder(WorkflowScheduler.sequential())` runs them one at a time. Both report
workflow time as the current time, so delays and periodic tasks (`Flowable.delay`, `interval`,
`timeout`) follow workflow time, and both replay deterministically. Note that ADK's default
`ToolExecutionMode.PARALLEL` still runs activity-backed tools one after another, because each call
blocks the workflow thread until its activity completes; set
`RunConfig.toolExecutionMode(PARALLEL_SUBSCRIBE)` to run them together.

Always build runners used in a workflow with `TemporalAdk.runnerBuilder()`. A runner built with
`Runner.builder()` alone fans out on ordinary threads, where model calls fail and the run never
completes.

### Tools

Plain ADK function tools run on the workflow thread. They must be deterministic and free of I/O:
no clocks, randomness, network, or files. They see ADK's live `ToolContext`, so they can read and
write session state, request confirmation, or start child workflows and timers through the
`Workflow` API.

To run a tool as an activity, give the agent an activity stub method:

```java
WeatherActivities weather =
    Workflow.newActivityStub(
        WeatherActivities.class,
        ActivityOptions.newBuilder().setStartToCloseTimeout(Duration.ofSeconds(30)).build());
LlmAgent agent =
    LlmAgent.builder()
        .name("assistant")
        .model(TemporalModel.create("gemini-2.5-flash"))
        .tools(ActivityTools.create(weather, "getWeather"))
        .build();
```

`ActivityTools.create` builds an ADK `FunctionTool` from the method declared on the
`@ActivityInterface`, so its name, description, and parameter schema come from the interface and
its `@Schema` annotations, and each call executes the activity. As with any ADK function tool,
every parameter needs a `@Schema(name = ...)` annotation unless the interface is compiled with
`-parameters`. `createLongRunning` marks the tool as long-running; a `requireConfirmation`
overload gates it behind a human decision. An activity that fails after its retries does not
report the error to the model: the run fails with the `ActivityFailure` and its typed cause, so
the workflow decides what to do.

### MCP toolsets

The workflow never connects to an MCP server. Register the server on the worker and reference it
by name in the agent:

```java
// Worker
GoogleAdkPlugin plugin =
    GoogleAdkPlugin.newBuilder()
        .mcpToolset(
            "docs",
            () ->
                new McpToolset(
                    StdioServerParameters.builder()
                        .command("npx")
                        .args(List.of("-y", "@modelcontextprotocol/server-filesystem", "/docs"))
                        .build()
                        .toServerParameters()))
        .build();

// Workflow
LlmAgent agent =
    LlmAgent.builder()
        .name("assistant")
        .model(TemporalModel.create("gemini-2.5-flash"))
        .tools(TemporalMcpToolset.create("docs"))
        .build();
```

`TemporalMcpToolset` is an ADK `BaseToolset`. Listing the tools runs the
`googleadk.ListMcpTools` activity and returns each tool's full declaration, including its
parameter schema; calling one runs `googleadk.CallMcpTool`. The worker creates each toolset once,
on first use, and closes it when the worker factory shuts down, after any in-flight call has
completed; calls that arrive during shutdown fail with a retryable error and move to another
worker. Any ADK `BaseToolset` can be registered, not only MCP ones. ADK's own `McpTool` retries a
failed server call three times before the activity attempt fails, on top of Temporal's retry
policy.

ADK asks a toolset for its tools before every model call, so by default each model call is
preceded by a list activity and the workflow always sees the server's current tools.
`TemporalMcpToolset.newBuilder("docs").cacheTools(true)` lists once per workflow execution instead.
`toolFilter(...)` or `toolNames(...)` narrow the tools, `activityOptions(...)` sets timeouts and
retries, and `requireConfirmation(true)` makes every call wait for a human decision.

### Human in the loop

ADK pauses a run when a tool requires confirmation or when a long-running tool returns nothing.
`Hitl` turns those pauses into a Temporal signal or update exchange:

```java
List<Event> events = TemporalAdk.run(runner, userId, sessionId, question);
List<Hitl.PendingConfirmation> pending = Hitl.pendingConfirmations(events);
if (!pending.isEmpty()) {
  Workflow.await(() -> decision != null); // set by a signal handler
  events =
      TemporalAdk.run(
          runner,
          userId,
          sessionId,
          Hitl.confirmationResponse(
              new Hitl.ConfirmationDecision(pending.get(0).functionCallId(), decision)));
}
```

The next turn goes through ADK's own confirmation flow: a confirmed tool runs, a rejected one
reports the rejection to the model. Long-running tools pause the run too: one that returns
nothing ends the model's turn at once, and one that returns an interim status gets a model reply
first; either way the call stays pending until `Hitl.functionResponse` delivers its result with
the next turn, and `Hitl.longRunningCalls` finds those calls.

### Sessions and continue-as-new

Temporal history is the durable store: the session service lives in the workflow, and a replay
rebuilds it. A new run after continue-as-new starts empty, so export the session before continuing
and import it in the next run:

```java
Session snapshot = Sessions.exportSession(runner.sessionService(), "chat", userId, sessionId);
Workflow.continueAsNew(new ConversationInput(snapshot, remainingMessages));
// in the next run
Sessions.importSession(runner.sessionService(), input.session());
```

The snapshot keeps the session state and events, including pending confirmation requests.
App-scoped (`app:`), user-scoped (`user:`), and temporary (`temp:`) state is not carried over,
and the imported state is exactly the snapshot's, whatever the replayed event deltas say. State
written by activities or MCP tools on the worker does not reach the session; return it as a tool
result instead.

### Payloads and failures

Workflow inputs, results, signals, queries, and activity payloads may contain ADK and GenAI types
(`LlmRequest`, `Event`, `Session`, `Content`, and so on), on their own or inside your own classes.
The plugin replaces the client's `json/plain` payload converter with `GoogleAdkPayloadConverter`:
Temporal's default Jackson configuration for your own types, plus the settings ADK's types need
(absent optionals omitted, unknown properties ignored, and a request's transient tool instances
skipped) applied to ADK and GenAI types only, so your other payloads keep Temporal's exact wire
format. The SDK's own `DefaultDataConverter.STANDARD_INSTANCE` is never modified: a client that
relies on it gets a new default converter with ADK support. A `DefaultDataConverter` your
application created, whether set on the client or registered with `GlobalDataConverter.register`,
is extended in place, because the SDK offers no way to copy one, so do not share that instance
with clients that should not see ADK types. If your client uses
a codec or another non-default converter, build it around
`GoogleAdkDataConverter.newDefaultInstance()` and set `customizeDataConverter(false)` on the
plugin; the plugin fails at client construction otherwise. If you register Jackson modules of your
own, construct `GoogleAdkPayloadConverter` from your mapper and pass the converter to
`DefaultDataConverter.withPayloadConverterOverrides`.

Model failures reach the workflow as `ActivityFailure` with an `ApplicationFailure` cause of type
`googleadk.ModelError` (`FailureTypes.MODEL_ERROR`). GenAI API errors with status 408, 409, 429, or
5xx are retryable; other 4xx responses, missing models, and factory errors are not. The status is
found through the wrappers RxJava and futures add around the error, so ADK's Gemini model, which
raises it from a future, is classified the same way. A response the model reports as an error (for
example a safety block) is not a failure: ADK emits it as a final event with an error code and no
text, exactly as outside Temporal. MCP failures use `googleadk.McpError`: an unknown toolset or tool
is non-retryable, a failing server call is retried, while a call the tool rejects outright (for
example because of a null argument value) is not. Model and MCP calls run on a background thread,
including the work a model does before it returns a flow (ADK's Claude model and its non-streaming
LangChain4j model block there), a toolset factory, and the session start `McpToolset` performs on
subscription, so with a heartbeat timeout on the activity options the activities heartbeat while
they run and return as soon as the server cancels them. The background call is left to finish on
its own and its result is dropped: an interrupt would not stop the request of any ADK model
(Gemini's completes on its HTTP client's own thread), and ADK's MCP classes answer one by retrying
on a new session, which would run a cancelled tool call twice. An MCP toolset is released only after
its last background call has finished, and a stdio server's process with it; each toolset is
released on its own, so a server that never answers keeps only its own toolset alive. Errors thrown
by tools that run on the workflow thread follow ADK's contract and go back to the model.

Exceptions ADK itself raises on the workflow thread (for example `LlmCallsLimitExceededException`
when `RunConfig.maxLlmCalls` is exceeded, or a `SessionException`) propagate out of
`TemporalAdk.run` unchanged. Temporal retries the workflow task on such exceptions; to fail the
workflow instead, list the types in
`WorkflowImplementationOptions.setFailWorkflowExceptionTypes` when registering the workflow, or
catch them and throw an `ApplicationFailure`.

Every model call carries the conversation so far, so a long session grows each `InvokeModel`
payload; Temporal limits a payload to 2 MB. Keep sessions bounded with ADK's event compaction or
by continuing as new with a session snapshot.

### Testing

`io.temporal.googleadk.testing.ScriptedModel` is a deterministic `BaseLlm` for tests: register it
on the plugin under the model name your agents use, run workflows against
`TestWorkflowEnvironment`, then replay the recorded history with a worker created from a client
that has the plugin. See this module's tests for exact activity sequences, parallel fan-out,
MCP, confirmation, continue-as-new, and determinism (a forced replay on a fresh worker)
examples.

## Limitations

- No token-level streaming out of the workflow and no live (bidirectional) mode.
- Tools that run on the workflow thread must be deterministic; ADK's code executor, memory, and
  artifact services that perform I/O are not supported inside workflows.
- ADK's OpenTelemetry tracing is global. Spans are no-ops without an installed SDK; with one
  installed, they are emitted again on every replay.
- `App` resumability and event compaction are ADK experimental features; they work with the
  runner builder returned here but are not covered by this integration's tests.
- Gemini context caching is inactive through the plugin: a request's cache configuration and
  metadata are not part of its JSON, so they do not reach the worker.

## Development

```bash
./gradlew spotlessCheck test stageDist
./gradlew resolveAndLockAll --write-locks            # after changing dependencies
./gradlew -PdependencyMode=latest resolveAndLockAll --write-locks --refresh-dependencies
```

The build uses Google Java Format through Spotless, strict dependency locking, and JUnit 5 with
Temporal's test environment; no network access or provider credentials are needed.

This integration is licensed under the MIT license; see the `LICENSE` file.
