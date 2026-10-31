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
and, for each MCP server name, the same factory function and idle timeout, as
when both are built from one options object. Two equal object literals are
refused.

Development details are in [typescript/README.md](../README.md).
