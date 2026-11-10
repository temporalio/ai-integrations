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

File data given as a `Uint8Array` crosses the Activity boundary as a base64
string, because Temporal's JSON payload converter does not keep `Uint8Array`
values. This applies to prompt file parts and to files the model generates. A
provider receives `{ type: 'data', data: '<base64>' }` in place of the bytes, and
`result.files` gives the same bytes back through `uint8Array`.

This applies to model Activities that a Workflow schedules after the upgrade. The
commands a Workflow sends do not change, so an execution that is in flight replays
as before. An execution that sent or received `Uint8Array` file data before the
upgrade keeps the corrupted bytes in its Event History. To repair it, reset it to
the `WorkflowTaskCompleted` event immediately before the `ActivityTaskScheduled`
event of the affected model Activity, for example with
`temporal workflow reset --event-id <id>`. The Workflow then schedules that
Activity again, with the bytes intact. A restart of the Workflow has the same
effect. Other executions need no action.

Prompts can include image and file parts given as URLs. The Workflow does not
download them, because the Workflow sandbox has no network. The model Activity
passes each URL that the Worker's model lists in `supportedUrls` to the model
and downloads the others with `createDownload()` from `ai`, as `generateText`
does. That download refuses private and local addresses. To change it, pass a
`download` function to `AiSdkPlugin`. A downloaded image gets the media type
that its bytes show, as in `generateText`. Downloaded bytes are not recorded in
Event History, and an Activity retry downloads them again. For that reason a
model call that downloaded a file returns no `request.body`. An
`experimental_download` function given to `generateText` runs in the Workflow
and is told that the model supports every URL.

This change is behind a Workflow patch. An execution that started on an earlier
version replays as it ran then, so an Activity that its `experimental_download`
function scheduled still matches its Event History. The new behavior starts
with the first model call that is not a replay.

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
set one in the provider options. While they wait on the provider, they heartbeat
at half that timeout, so a lost Worker is noticed, and a cancellation request
reaches the Activity, within about that time. Set `heartbeatTimeout: 0` to turn
this off. Then a lost Worker is noticed only at `startToCloseTimeout`, and a
cancellation request does not reach a call that is waiting on the provider.

When `invokeModelStreaming` is retried, the new attempt publishes all of its
stream parts to the topic again, after the parts the failed attempt published.
A subscriber that wants only the last attempt can discard what it has when a
`stream-start` part arrives. The `@ai-sdk/openai` models send `stream-start` as
the first part of every call, but the provider specification does not require
it, so a custom provider may not send it.

## Event History size

The model Activities leave the raw HTTP bodies out of their results, because
Event History records every result. The request body repeats the prompt that the
Activity input already holds, and the response body repeats the content. So
`request.body` and `response.body` are `undefined` in Workflow code. This holds
for `generateText` and `streamText` results even when `include.requestBody` or
`include.responseBody` is set, for `embed` and `embedMany` responses, and for
middleware that reads the model result. Response headers, `id`, `modelId` and
`timestamp` are kept.

By default `embedMany` sends all values to one `invokeEmbeddingModel` Activity,
and Temporal rejects a result over 2 MB (about 70 to 100 embeddings of 1536
dimensions). Set `maxEmbeddingsPerCall` in the `embeddingModel` options of
`TemporalProvider` to split the values into one Activity per chunk. Changing
this value in a deployed Workflow needs Workflow versioning, because it changes
the Activities the Workflow schedules.

The split applies to the values only. Each chunk gets the same `providerOptions`
as the full call. So do not set `maxEmbeddingsPerCall` together with provider
options that hold one entry for each value, such as
`providerOptions.google.content`. The provider refuses such a chunk, because the
counts do not match. For those calls, split the values in the Workflow and call
`embedMany` one time for each chunk, with the options of that chunk.

```typescript
const provider = new TemporalProvider({ embeddingModel: { maxEmbeddingsPerCall: 32 } });
```

## Abort signals and timeouts

An `abortSignal` passed to `generateText`, `streamText` or `embedMany` stays in
the Workflow. When it aborts, the plugin cancels the model Activity and the call
rejects with the signal's reason. An already aborted signal schedules no
Activity. The Activity passes its own cancellation signal to the model, but a
running Activity learns about cancellation only through heartbeats. The AI SDK
`timeout` option does not work in the Workflow sandbox. A number or `totalMs`
fails with `AbortSignal is not defined`, and `stepMs` fails with
`DOMException is not defined` when its timer fires. Use the Activity
`startToCloseTimeout` instead.

Development details are in [typescript/README.md](../README.md).
