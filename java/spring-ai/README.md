# Temporal Spring AI

Public Preview integration for Spring AI 2.1 and Spring Boot 4.2. Model calls
execute as Temporal Activities; Activity, Nexus, side-effect, and deterministic
tools execute through the ChatClient tool-calling advisor on the workflow thread.

The new coordinate is `io.temporal:spring-ai` on
[Maven Central](https://central.sonatype.com/artifact/io.temporal/spring-ai).
CI tests Java 17 and 25; published bytecode targets Java 17.

## Requirements and installation

- Java 17 or later.
- Spring AI 2.1.0 or later in the 2.x series on both the Workflow and Activity workers.
- Spring Boot 4.2.0 or later in the 4.x series.
- Temporal Java SDK and `temporal-spring-boot-starter` 1.40.0 or later.
- A Spring AI provider starter for your chosen model. This integration supplies no provider.

Install the standalone integration and import the matching Spring AI BOM in your application.

```groovy
implementation 'io.temporal:spring-ai:<version>'
implementation 'io.temporal:temporal-spring-boot-starter:1.40.0'
implementation 'io.temporal:temporal-sdk:1.40.0'
```

Inside a workflow, construct the client with `TemporalChatClient.builder(ActivityChatModel.forDefault())`
and register tools with `defaultTools(...)`. The entry points `TemporalChatClient`,
`ActivityChatModel`, and Temporal tool annotations retain their names and packages.
Model calls run as Activities; tool calls remain durable Temporal operations.

## Upgrading from earlier releases

Spring AI 2.0 applications can use `io.temporal:spring-ai:0.1.1`. Spring AI 1 and
Boot 3 applications can use `io.temporal:temporal-spring-ai:1.40.x`. Spring AI 2
uses immutable options and Jackson 3. Pass option builders
to ChatClient, for example `.defaultOptions(OpenAiChatOptions.builder().reasoningEffort("high"))`.
`ActivityChatModel.call(...)` returns a raw model response; use ChatClient's advisor
loop to execute tools. The removed `internalToolExecutionEnabled` flag is unnecessary.

This release changes the chat Activity payload format. Keep in-flight workflows on
their original workers until they finish. Route new workflows to a separate Worker
Deployment Version running this release.

Streaming and tool context remain unsupported. Provider-specific options survive
Activity serialization through their public builders; callbacks are transported as
definitions and execute only in the workflow. Retry, per-model Activity options,
response metadata, and media-size guards retain their behavior. Optional vector-store,
embedding, and MCP modules are detected through Spring auto-configuration.

## Ordered chat content

With Spring AI 2.1, the chat Activity preserves the order of message parts and
their provider-owned payloads through tool continuations. For example, Anthropic
thinking signatures and redacted thinking survive the round trip alongside text
and tool calls.

Large chat prompts and model responses, including ordered parts and tool results,
can exceed Temporal's payload limit. Pass references for large media. Supply media
as bytes or URI strings. Spring AI 2.1's media codec converts `java.net.URL` objects
to strings, so the integration refuses that input type. When creating media in
Workflow code, set a deterministic name with `Media.builder().name(...)`. Spring AI
otherwise generates a name with a random UUID. The integration preserves media names
verbatim, including names returned by a model Activity. For a unique name in a
Workflow, build it with `Workflow.randomUUID()`.

User, system, and tool message metadata supports string keys with strings, booleans,
`Integer` values, finite `Double` values, nulls, and nested lists and maps of those
types. The chat call fails before scheduling the Activity if this metadata contains
an unsupported value. Convert bytes, URLs, and dates to strings before attaching
them as metadata. Provider-owned continuation data travels in the message parts.

## Structured output

`entity(...)` and `responseEntity(...)` convert the model's answer in workflow code,
after the model Activity has completed. When a provider refusal or safety filter
leaves no usable answer, Spring AI throws `ContentFilteredException`. When
`BeanOutputConverter` fails to convert an answer, it raises a Jackson
`tools.jackson.core.JacksonException`. If the workflow doesn't handle it, the
workflow task fails and is retried, but the model's answer is already in the
history, so every retry fails the same way and the workflow stays running.

Catch the exception around the call, for example to ask the model again or return
a fallback, or make it fail the workflow so a workflow retry policy can start
another run:

```java
WorkflowImplementationOptions options =
    WorkflowImplementationOptions.newBuilder()
        .setFailWorkflowExceptionTypes(JacksonException.class)
        .build();
worker.registerWorkflowImplementationTypes(options, MyWorkflowImpl.class);
```

## Provider retries

For OpenAI chat Activities, native OpenAI HTTP errors retain their status, error
body, and diagnostic response headers (`retry-after`, `retry-after-ms`,
`x-request-id`, `x-ratelimit-*`, and `x-should-retry`) in `ApplicationFailure`
details. `insufficient_quota` responses are always non-retryable. Otherwise,
`X-Should-Retry: true` or `false` overrides the HTTP status. Without a recognized
directive, rate limits (429), request timeouts (408), conflicts (409), and server
errors are retryable; other HTTP client errors are non-retryable.

`Retry-After` (seconds or an HTTP date) and `retry-after-ms` set Temporal's next
retry delay; missing or invalid delays leave the Activity's configured backoff
in effect. A positive provider delay overrides `maximumInterval`, which does
not cap it. Maximum attempts and schedule-to-close timeouts still apply. To
exclude additional errors with `RetryOptions.Builder.setDoNotRetry(...)`, use
the exact failure type, such as `com.openai.errors.RateLimitException`.

Disable provider SDK retries on models registered with Temporal so each Activity
attempt makes one provider request. For a manually constructed OpenAI model, use
`OpenAiChatOptions.builder().maxRetries(0)` when building the worker's model.
For Spring Boot's OpenAI chat model:

```properties
spring.ai.openai.chat.max-retries=0
```

Set these options when constructing the provider client. Use a separate client
for calls outside Temporal that need SDK retries.

## Development

```bash
./gradlew spotlessCheck test stageDist
```

The Java/OS matrix uses the committed dependency lock. Update it with
`./gradlew resolveAndLockAll --write-locks`.

Nightly CI and manual runs with `latest-deps=true` test stable releases in Temporal
1.x, Spring AI 2.1 through the latest 2.x release, and Spring Boot 4.2 through
the latest 4.x release.
To run that check locally:

```bash
./gradlew -PdependencyMode=latest resolveAndLockAll --write-locks --refresh-dependencies
./gradlew -PdependencyMode=latest spotlessCheck test stageDist
```

Latest mode writes a separate ignored lockfile and reuses its selected versions
through testing and building. Ordinary CI uses the committed locks.

The implementation and tests use the repository's MIT license.
