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
each MCP server. If the Worker already has a different Activity with one of
these names (your own Activity, another integration's MCP Activity, or an
`AiSdkPlugin` with different options), `Worker.create` fails with
`Duplicate registration of Activity '<name>'`. Earlier versions kept the last
one without a warning. Rename the other Activity or MCP server.

Development details are in [typescript/README.md](../README.md).
