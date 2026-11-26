# Temporal LangChain4j

Pre-release integration between [LangChain4j](https://github.com/langchain4j/langchain4j) and
Temporal. A LangChain4j AI Service built inside a Temporal workflow runs every model call, streamed
or not, and every MCP tool call as a Temporal Activity, so a run survives worker restarts, retries
failed calls, and replays deterministically instead of calling the model again.

APIs and configuration may change before General Availability.

The coordinate is `io.temporal:langchain4j`. CI tests Java 17 and 25; published bytecode targets
Java 17.

## Requirements and installation

- Java 17 or later.
- LangChain4j 1.23.0 or later (`langchain4j-core`; add `langchain4j` for AI Services).
- Temporal Java SDK 1.40.0 or later. The integration compiles against the SDK but does not pull
  it in; your application supplies it.
- Optional: `dev.langchain4j:langchain4j-mcp` (1.23.0-beta33 or later) for [MCP tools](#mcp-tools),
  and `io.temporal:temporal-workflowstreams` at your SDK's version for
  [live model output](#live-model-output). The integration compiles against both but pulls in
  neither.

```groovy
implementation 'io.temporal:langchain4j:0.0.1'
implementation 'io.temporal:temporal-sdk:1.40.0'
implementation 'dev.langchain4j:langchain4j:1.23.0'
```

## Hello world

Register the provider's `ChatModel` on the worker under a name. Turn off the provider client's
own retries (`maxRetries(0)`), so that each Activity attempt sends one request and Temporal owns
retries:

```java
LangChain4jPlugin plugin =
    LangChain4jPlugin.newBuilder()
        .chatModel(
            "openai",
            () ->
                OpenAiChatModel.builder()
                    .apiKey(System.getenv("OPENAI_API_KEY"))
                    .modelName("gpt-4o-mini")
                    .maxRetries(0)
                    .build())
        .build();
WorkflowClient client =
    WorkflowClient.newInstance(
        service, WorkflowClientOptions.newBuilder().setPlugins(plugin).build());
```

Workers created from that client register the plugin's Activities, including `langchain4j.Chat`.
In the workflow, build
the AI Service with a `TemporalChatModel` that names the registration:

```java
public class AssistantWorkflowImpl implements AssistantWorkflow {
  @Override
  public String ask(String question) {
    Assistant assistant =
        AiServices.builder(Assistant.class)
            .chatModel(TemporalChatModel.create("openai"))
            .tools(new Calculator())
            .chatMemory(MessageWindowChatMemory.withMaxMessages(20))
            .build();
    return assistant.chat(question);
  }
}
```

The workflow only carries the registration name. The worker runs the request against the
registered model, so API keys and provider clients stay on the worker.

## What the integration gives you

### Model calls as Activities

`TemporalChatModel` implements LangChain4j's `ChatModel`. Each `chat` call schedules one
`langchain4j.Chat` Activity with the summary `Chat: <name>`; set
`TemporalChatModel.newBuilder(name).activityOptions(...)` for timeouts and retries. The default is
a two-minute start-to-close timeout and the SDK's default retry policy.

The worker's registered model applies its own default request parameters and listeners, exactly as
when it is called directly. Declare the registered model's capabilities on the workflow side with
`supportedCapabilities(...)`: with `Capability.RESPONSE_FORMAT_JSON_SCHEMA`, an AI Service that
returns a POJO or record sends the JSON schema to the model.

### Tools

`@Tool` methods of objects passed to `AiServices.builder(...).tools(...)` run on the workflow
thread, between model calls, as LangChain4j runs them. Keep them deterministic: they may read
workflow state and call Temporal APIs, such as an Activity stub, but must not do I/O directly.

### Streaming

`TemporalStreamingChatModel` implements LangChain4j's `StreamingChatModel`, so an AI Service method
that returns a `TokenStream` works in a workflow. Register the provider's `StreamingChatModel` on
the worker and name it in the workflow:

```java
// Worker
LangChain4jPlugin.newBuilder()
    .streamingChatModel(
        "openai",
        () -> OpenAiStreamingChatModel.builder()
            .apiKey(System.getenv("OPENAI_API_KEY"))
            .modelName("gpt-4o-mini")
            .build())
    .build();

// Workflow
StreamingAssistant assistant =
    AiServices.builder(StreamingAssistant.class)
        .streamingChatModel(TemporalStreamingChatModel.create("openai"))
        .tools(new Calculator())
        .build();
List<String> tokens = new ArrayList<>();
List<ChatResponse> responses = new ArrayList<>();
List<Throwable> errors = new ArrayList<>();
assistant
    .chat(question)
    .onPartialResponse(tokens::add)
    .onCompleteResponse(responses::add)
    .onError(errors::add)
    .start();
```

Each call schedules one `langchain4j.StreamingChat` Activity with the summary
`StreamingChat: <name>`. The worker streams the request and returns the partial responses,
partial thinking, partial and complete tool calls, and the complete response. When the Activity
completes, the workflow passes them to the handler in the order the model produced them, then the
complete response, all on the workflow thread, so `start()` has run every callback when it returns
and replay gives the same callbacks without calling the model. Tools run between model calls, as
with `TemporalChatModel`. A failed Activity is passed to `onError`.

The partial output reaches the workflow only when the Activity completes. Cancelling the
`StreamingHandle` from a `...WithContext` callback stops the remaining callbacks, including the
complete response, but the model has already finished. To show output while the model runs, see
[Live model output](#live-model-output).

### Live model output

The streaming model Activity can also publish each partial output to a Workflow Stream topic
while it runs. Add the experimental `temporal-workflowstreams` module at the same version as your
SDK; this path needs the unshaded `temporal-sdk` artifact, and the other Activities keep working
with the shaded one.

```groovy
implementation 'io.temporal:temporal-workflowstreams:1.40.0'
```

Host a `WorkflowStream` in the workflow and set the model's `streamingTopic`. Parts are published
in 100 ms batches and still returned as the Activity result. Subscribers read the stream through an
Update, which a closed workflow can't serve, so publish a completion marker and give clients a
bounded window to acknowledge it:

```java
public class LiveChatWorkflowImpl implements LiveChatWorkflow {
  private final WorkflowStream output;
  private boolean consumed;

  @WorkflowInit
  public LiveChatWorkflowImpl(String message) {
    output = WorkflowStream.newInstance();
  }

  @Override
  public String chat(String message) {
    StreamingAssistant assistant =
        AiServices.create(
            StreamingAssistant.class,
            TemporalStreamingChatModel.newBuilder("openai").streamingTopic("model-output").build());
    StringBuilder answer = new StringBuilder();
    assistant.chat(message).onPartialResponse(answer::append).onError(e -> {}).start();
    output.topic("turn-status").publish("complete");
    Workflow.await(Duration.ofSeconds(10), () -> consumed);
    return answer.toString();
  }

  @Override // a @SignalMethod on the workflow interface
  public void streamConsumed() {
    consumed = true;
  }
}
```

Items are JSON written by the standard payload converter (the client's codec applies once, to the
publish Signal); decode them as `ModelStreamEvent`:

```java
try (WorkflowStreamClient stream = WorkflowStreamClient.newInstance(client, workflowId);
    WorkflowStreamSubscription subscription =
        stream.subscribe(
            SubscribeOptions.newBuilder()
                .setTopics("model-output", "turn-status")
                .setFromOffset(0)
                .build())) {
  for (WorkflowStreamItem item : subscription) {
    if (item.getTopic().equals("turn-status")) {
      workflow.streamConsumed();
      break;
    }
    ModelStreamEvent event =
        DefaultDataConverter.STANDARD_INSTANCE.fromPayload(
            item.getPayload(), ModelStreamEvent.class, ModelStreamEvent.class);
    if (event.kind() == ModelStreamEvent.Kind.PART) {
      System.out.print(event.part().text());
    }
  }
}
```

A `ModelStreamEvent` carries the Workflow ID, Run ID, Activity ID, attempt, and a zero-based
sequence; a retried Activity keeps its ID, increments the attempt, and restarts the sequence. `PART`
wraps one partial response, thinking or tool call. `COMPLETE`, `FAILED` and `CANCELED` close one
attempt and are best effort: they can be missing after a worker or transport failure, and `COMPLETE`
does not commit the Activity result. Treat the workflow result or your own marker as the final
outcome.

Each publish Signal waits at most five seconds. A publishing failure fails the attempt with
`LangChain4jStreamError`. A batch the server refuses as invalid or too large fails without
retrying the model; other transport failures, such as rate limiting and outages, follow the
Activity's retry policy. A missing or mismatched `temporal-workflowstreams` artifact fails without
retries. Continue-As-New before reaching the server's Signal limit, and check Run IDs across
Continue-As-New: publication targets the current Run of the Workflow ID. This follows the Google
ADK integration's live output.

### MCP tools

The workflow never connects to an MCP server. Register LangChain4j's MCP client on the worker and
give `TemporalMcpClient` to LangChain4j's own `McpToolProvider` in the workflow:

```java
// Worker
LangChain4jPlugin.newBuilder()
    .mcpClient(
        "docs",
        () -> DefaultMcpClient.builder()
            .key("docs")
            .transport(
                StdioMcpTransport.builder()
                    .command(
                        List.of("npx", "-y", "@modelcontextprotocol/server-filesystem", "/docs"))
                    .build())
            .build())
    .build();

// Workflow
Assistant assistant =
    AiServices.builder(Assistant.class)
        .chatModel(TemporalChatModel.create("openai"))
        .toolProvider(
            McpToolProvider.builder().mcpClients(TemporalMcpClient.create("docs")).build())
        .build();
```

`TemporalMcpClient` implements LangChain4j's `McpClient`. Listing the tools runs the
`langchain4j.ListMcpTools` Activity and returns each tool's specification, including its parameter
schema; calling one runs `langchain4j.CallMcpTool`. `McpToolProvider` asks for the tools on every
AI Service call, so each call starts with a list Activity and the workflow sees the server's
current tools. The provider's filters, tool name mapping and `failIfOneServerFails` work unchanged,
because they are LangChain4j's own code. `TemporalMcpClient.newBuilder("docs").activityOptions(...)`
sets timeouts and retries; the default is a one-minute start-to-close timeout and the SDK's default
retry policy.

The worker creates each client once, on first use, and closes it when the worker factory shuts
down, after any in-flight call has completed; calls that arrive during shutdown fail with a
retryable error and move to another worker. Tool calls use the client's cancellable
`executeToolAsync` when it has one (LangChain4j's `DefaultMcpClient` does), so a cancelled Activity
with a heartbeat timeout also cancels the call on the server.

Tool errors reach the AI Service the way LangChain4j's MCP client reports them, so its tool error
handlers apply:

- An error result the server wrote for the model (`isError`) completes the Activity and is thrown
  in the workflow as LangChain4j's `McpApplicationErrorException`, which AI Services send to the
  model.
- A rejection of the arguments (JSON-RPC invalid params) fails the Activity without retries, with
  the type `LangChain4jMcpToolArgumentsError`, and is thrown as `ToolArgumentsException`.
- Any other failure, such as a server error or a broken connection, follows the Activity's retry
  policy with the type `LangChain4jMcpError`; once it gives up, it is thrown as
  `ToolExecutionException`. A cancellation is rethrown as it is.

As for model calls, these failures name only the client, the tool and the exception class; the
server's message is logged on the worker and kept out of the history. A workflow that names a
client the worker doesn't serve fails with the non-retryable type `LangChain4jUnknownMcpClient`.

### Chat memory

A `MessageWindowChatMemory` (or `TokenWindowChatMemory`) created in the workflow holds the
conversation in workflow state. Replay rebuilds it from the model Activities' results, so no
external `ChatMemoryStore` is needed.

### Structured output

An AI Service method that returns a POJO, a record or another non-text type converts the model's
answer in workflow code, after the model Activity has completed. When the answer doesn't match the
return type, LangChain4j throws `OutputParsingException`. If the workflow doesn't handle it, the
workflow task fails and is retried, but the model's answer is already in the history, so every
retry fails the same way and the workflow stays running.

Catch the exception around the call, for example to ask the model again or return a fallback, or
make it fail the workflow so a workflow retry policy can start another run:

```java
WorkflowImplementationOptions options =
    WorkflowImplementationOptions.newBuilder()
        .setFailWorkflowExceptionTypes(OutputParsingException.class)
        .build();
worker.registerWorkflowImplementationTypes(options, MyWorkflowImpl.class);
```

### Payloads

The Activity input and result are built from LangChain4j's own JSON forms: messages are written by
`ChatMessageSerializer`, tool specifications by `ToolSpecification.toJson()`, and a response
format's JSON schema by LangChain4j's schema mapper. Every content type they support, such as
images given as base64 data or URLs, crosses the Activity boundary unchanged.

The input carries the whole conversation, so a long conversation or inline images can exceed
Temporal's payload size limit. Prefer image URLs over base64 data, and bound the chat memory. The
`langchain4j.StreamingChat` result carries the partial output as well as the complete response, so
it can be up to about twice the size of a `langchain4j.Chat` result. A `langchain4j.CallMcpTool`
result holds the tool's whole result, including any images, and a `langchain4j.ListMcpTools` result
every tool specification of the server; filter large servers' tools with `McpToolProvider`.

MCP tool results use LangChain4j's message JSON for their contents; the structured result and the
attributes travel as JSON and come back as maps, lists and JSON values.

### Failures

A provider error that LangChain4j marks as final (a `NonRetriableException`, such as
`AuthenticationException` or `InvalidRequestException`) fails the Activity without retries; for a
streaming model, this includes an error passed to the handler's `onError`. The
Activity's retry policy decides for every other error, such as `RateLimitException`. The failure
type is `LangChain4jModelError`. Its message names only the model and the exception class: the provider's message, stack trace and causes can hold request data, so they're logged on the worker and kept out of the workflow history. A workflow that names a model the worker doesn't serve fails with
the non-retryable type `LangChain4jUnknownModel`.

When the Activity has a heartbeat timeout, it heartbeats every third of that timeout while the
model or MCP call runs, so a dead worker is detected early. A heartbeat that reports a cancellation
also cancels a streaming model's stream, through its `StreamingHandle` when the provider supports
that, and an MCP tool call.

## Limitations

- LangChain4j's AI Services give each call an `InvocationContext` whose `invocationId()` is a random UUID and
  whose `timestamp()` is the wall-clock time, and offer no way to supply them. Inside a workflow they differ on
  replay. Model calls and tools don't depend on them, so replay stays deterministic, but workflow code, tools and
  listeners must not branch on them or pass them to Temporal APIs. Use `Workflow.randomUUID()` and
  `Workflow.currentTimeMillis()` instead.
- Provider-specific request parameters, such as `OpenAiChatRequestParameters`, are refused with a
  clear error: only the provider's model understands them, and it runs on the worker. Set them on
  the registered model instead. The workflow receives the provider-independent response metadata
  (id, model name, token usage and finish reason); provider-specific metadata is dropped.
- `ChatModel.chat` has no cancellation handle, so an Activity cancellation stops the heartbeats
  but not a request that is already in flight.
- AI Services that return a `CompletableFuture` or a reactive `Flow.Publisher`, and
  `StreamingChatModel.chat(ChatRequest)` returning a `Publisher`, are not supported: they run on
  LangChain4j's executors, which can't run inside a workflow. Use a `TokenStream`.
- A streaming model's `onUnmappedRawEvent` callbacks carry provider-specific objects and are not
  passed to the workflow.
- `TemporalMcpClient` supports MCP tools only: resources (including `McpToolProvider`'s
  resources-as-tools), prompts, roots, subscriptions and health checks throw
  `UnsupportedOperationException`, and `executeToolAsync` is not available in workflows. The
  `InvocationContext` of a call is not sent to the worker, because its id and timestamp differ on
  replay. LangChain4j's MCP client applies its own timeouts on the worker; set
  `toolExecutionTimeout` below the Activity's start-to-close timeout.
- Not supported yet: RAG content retrievers, Activity-backed and side-effect tools declared with an
  annotation, and Spring Boot auto-configuration.
- `executeToolsConcurrently` is not supported: LangChain4j runs concurrent tools on its own
  executor, which can't run inside a workflow. Don't enable it on AI Services that run in a
  workflow.

## Development

```bash
./gradlew spotlessCheck test stageDist
```
