# Temporal Vercel AI SDK integration

Release stage: Generally Available.

Run Vercel AI SDK agents as durable Temporal Workflows. Model and MCP requests
execute as Activities so Temporal can record results, retry failures and replay
completed calls.

```typescript
// Worker setup
import { AiSdkPlugin } from '@temporalio/vercel-ai-sdk';
import { openai } from '@ai-sdk/openai';

const plugins = [new AiSdkPlugin({ modelProvider: openai })];
```

```typescript
// Workflow code
import { generateText } from 'ai';
import { temporalProvider } from '@temporalio/vercel-ai-sdk/workflow';

export async function answer(prompt: string): Promise<string> {
  const result = await generateText({
    model: temporalProvider.languageModel('gpt-4o-mini'),
    prompt,
  });
  return result.text;
}
```

Register `plugins` with the Temporal Worker. Provider credentials belong in the
Worker environment. See the [integration guide](https://docs.temporal.io/develop/typescript/integrations/ai-sdk)
for model, tool, streaming and MCP configuration. That guide currently uses the
SDK-owned `@temporalio/ai-sdk` name; this repository owns the new
`@temporalio/vercel-ai-sdk` coordinate.

Provider errors follow the AI SDK's own retry verdict. An `APICallError` with
`isRetryable: false`, for example a 400 or 401 response, fails the model
Activity with a non-retryable `ApplicationFailure`, and Temporal does not retry
it. The same applies to such an error in a streamed `error` part. The failure
type does not change. It is `APICallError` for a thrown error and `Error` for a
streamed `error` part. The Workflow receives an `ActivityFailure` whose `cause`
is that `ApplicationFailure`. Other errors, such as a 408, 409, 429 or 5xx response or
a lost connection, are retried under the Activity's retry policy.

The plugin registers Activities named `invokeModel`, `invokeModelStreaming` and
`invokeEmbeddingModel`, plus `<server>-listTools` and `<server>-callTool` for
each MCP server. A different Activity with one of these names now fails with
`Duplicate registration of Activity '<name>'`; earlier versions kept the last
one without a warning. If it is registered before `AiSdkPlugin` (in the Worker's
`activities` or by an earlier plugin), `Worker.create` fails. If a plugin
applied after `AiSdkPlugin` replaces one, `worker.run()` fails before the Worker
polls. `Worker.create` applies the Worker's `plugins` in order, then the
`NativeConnection`'s.

To fix a collision with your own Activity or an MCP server name, rename one of
them. `StrandsPlugin` (`@temporalio/strands-agents`) also registers
`invokeModel` and `invokeModelStreaming`, and neither plugin can rename them, so
run the two integrations on separate Workers and task queues. Two `AiSdkPlugin`
instances count as the same only when they share the `modelProvider` object
and, for each MCP server name both register, the same factory function and idle
timeout. A provider or factory created separately for each instance is refused
even when its settings are equal, for example `createOpenAI({ apiKey })` called
once per plugin, or an `async () => ...` factory written inline in each options
object.

## Heartbeats and streaming

The model and embedding Activities get a 10-second `heartbeatTimeout` unless you
set one in the provider options. They heartbeat every half of that timeout while
they run, so a lost Worker is noticed, and a cancellation request reaches the
Activity, within about that time. A call shorter than half the timeout sends no
heartbeat. Set `heartbeatTimeout: 0` to turn heartbeats off. Then a lost Worker
is noticed only at `startToCloseTimeout`, and a cancellation request does not
reach the Activity.

When `invokeModelStreaming` is retried, the new attempt publishes all of its
stream parts to the topic again, after the parts the failed attempt published.
A subscriber that wants only the last attempt can discard what it has when a
`stream-start` part arrives. The `@ai-sdk/openai` models send `stream-start` as
the first part of every call, but the provider specification does not require
it, so a custom provider may not send it.

Development details are in [typescript/README.md](../README.md).
