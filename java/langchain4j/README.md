# Temporal LangChain4j

Public Preview integration between [LangChain4j](https://github.com/langchain4j/langchain4j) and
Temporal. A LangChain4j AI Service built inside a Temporal workflow runs every model call as a
Temporal Activity, so a run survives worker restarts, retries failed calls, and replays
deterministically instead of calling the model again.

APIs and configuration may change before General Availability.

The coordinate is `io.temporal:langchain4j`. CI tests Java 17 and 25; published bytecode targets
Java 17.

## Requirements and installation

- Java 17 or later.
- LangChain4j 1.23.0 or later (`langchain4j-core`; add `langchain4j` for AI Services).
- Temporal Java SDK 1.40.0 or later. The integration compiles against the SDK but does not pull
  it in; your application supplies it.

```groovy
implementation 'io.temporal:langchain4j:0.1.0'
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

Workers created from that client register the `langchain4j.Chat` Activity. In the workflow, build
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

### Chat memory

A `MessageWindowChatMemory` (or `TokenWindowChatMemory`) created in the workflow holds the
conversation in workflow state. Replay rebuilds it from the model Activities' results, so no
external `ChatMemoryStore` is needed.

### Payloads

The Activity input and result are built from LangChain4j's own JSON forms: messages are written by
`ChatMessageSerializer`, tool specifications by `ToolSpecification.toJson()`, and a response
format's JSON schema by LangChain4j's schema mapper. Every content type they support, such as
images given as base64 data or URLs, crosses the Activity boundary unchanged.

The input carries the whole conversation, so a long conversation or inline images can exceed
Temporal's payload size limit. Prefer image URLs over base64 data, and bound the chat memory.

### Failures

A provider error that LangChain4j marks as final (a `NonRetriableException`, such as
`AuthenticationException` or `InvalidRequestException`) fails the Activity without retries. The
Activity's retry policy decides for every other error, such as `RateLimitException`. The failure
type is `LangChain4jModelError`. A workflow that names a model the worker doesn't serve fails with
the non-retryable type `LangChain4jUnknownModel`.

When the Activity has a heartbeat timeout, it heartbeats every third of that timeout while the
model call runs, so a dead worker is detected early.

## Limitations

- Provider-specific request parameters, such as `OpenAiChatRequestParameters`, are refused with a
  clear error: only the provider's model understands them, and it runs on the worker. Set them on
  the registered model instead. The workflow receives the provider-independent response metadata
  (id, model name, token usage and finish reason); provider-specific metadata is dropped.
- `ChatModel.chat` has no cancellation handle, so an Activity cancellation stops the heartbeats
  but not a request that is already in flight.
- Not supported yet: `StreamingChatModel`, RAG content retrievers, Activity-backed and
  side-effect tools declared with an annotation, MCP tools, and Spring Boot auto-configuration.
- `executeToolsConcurrently` is not supported: LangChain4j runs concurrent tools on its own
  executor, which can't run inside a workflow. Don't enable it on AI Services that run in a
  workflow.

## Development

```bash
./gradlew spotlessCheck test stageDist
```
