// Test workflow using AI model
// `ai` is deliberately imported before the workflow entry point: the bundle for these tests is
// built through AiSdkPlugin, whose `configureBundler` prepends the polyfill installer to the
// webpack entry, so import order must not matter.
import {
  embedMany,
  generateText,
  isStepCount,
  Output,
  streamText,
  tool,
  wrapLanguageModel,
} from 'ai';
import type { LanguageModelMiddleware } from 'ai';
import type { LanguageModelV4StreamPart } from '@ai-sdk/provider';
import { OpenTelemetry } from '@ai-sdk/otel';
import { z } from 'zod/v3';
import { defineSignal, proxyActivities, setHandler } from '@temporalio/workflow';
import { WorkflowStream } from '@temporalio/workflow-streams/workflow';
import {
  TemporalEmbeddingModel,
  TemporalLanguageModel,
  TemporalMCPClient,
  TemporalProvider,
  temporalProvider,
} from '../../workflow';
import type * as activities from '../activities/ai-sdk';

const { downloadNotes, getWeather } = proxyActivities<typeof activities>({
  startToCloseTimeout: '1 minute',
});

export async function helloWorldAgent(prompt: string): Promise<string> {
  const result = await generateText({
    model: temporalProvider.languageModel('gpt-4o-mini'),
    prompt,
    instructions: 'You only respond in haikus.',
  });
  return result.text;
}

export async function toolsWorkflow(question: string): Promise<string> {
  const result = await generateText({
    model: temporalProvider.languageModel('gpt-4o-mini'),
    prompt: question,
    instructions: 'You are a helpful agent.',
    tools: {
      getWeather: tool({
        description: 'Get the weather for a given city',
        inputSchema: z.object({
          location: z.string().describe('The location to get the weather for'),
        }),
        execute: async (input, _options) => getWeather(input),
      }),
    },
    stopWhen: isStepCount(5),
  });
  return result.text;
}

const transparentPng =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';

export async function imageToolWorkflow(): Promise<string> {
  if (btoa(atob(transparentPng)) !== transparentPng) {
    throw new Error('Base64 polyfill round trip failed');
  }

  const result = await generateText({
    model: temporalProvider.languageModel('gpt-4o-mini'),
    prompt: 'Call the screenshot tool, then describe the image.',
    tools: {
      screenshot: tool({
        description: 'Take a screenshot',
        inputSchema: z.object({}),
        execute: async () => ({ data: transparentPng, mimeType: 'image/png' }),
        toModelOutput: ({ output }) => ({
          type: 'content',
          value: [
            {
              type: 'file',
              data: { type: 'data', data: output.data },
              mediaType: output.mimeType,
            },
          ],
        }),
      }),
    },
    stopWhen: isStepCount(3),
  });
  return result.text;
}

export async function binaryFileWorkflow(bytes: number[]): Promise<number[]> {
  const result = await generateText({
    model: temporalProvider.languageModel('gpt-4o-mini'),
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Edit this file.' },
          {
            type: 'file',
            data: { type: 'data', data: new Uint8Array(bytes) },
            mediaType: 'application/octet-stream',
          },
        ],
      },
    ],
  });
  return Array.from(result.files[0]!.uint8Array);
}

export async function fileUrlWorkflow(baseUrl: string): Promise<string> {
  const result = await generateText({
    model: temporalProvider.languageModel('gpt-4o-mini'),
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Describe the image and the notes.' },
          { type: 'image', image: new URL(`${baseUrl}/image`) },
          { type: 'file', data: new URL(`${baseUrl}/notes`), mediaType: 'text/plain' },
        ],
      },
    ],
  });
  return result.text;
}

export async function mediaTypeUrlWorkflow(baseUrl: string): Promise<string> {
  const result = await generateText({
    model: temporalProvider.languageModel('gpt-4o-mini'),
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Describe the image.' },
          {
            type: 'file',
            data: new URL(`${baseUrl}/image`),
            mediaType: 'application/octet-stream',
          },
        ],
      },
    ],
  });
  return result.text;
}

// Downloads through an Activity of its own, which worked before the model Activity did it.
export async function experimentalDownloadWorkflow(baseUrl: string): Promise<string> {
  const result = await generateText({
    model: temporalProvider.languageModel('gpt-4o-mini'),
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Describe the notes.' },
          { type: 'file', data: new URL(`${baseUrl}/notes`), mediaType: 'text/plain' },
        ],
      },
    ],
    experimental_download: async (requests) =>
      await Promise.all(
        requests.map(async ({ url, isUrlSupportedByModel }) => {
          if (isUrlSupportedByModel) return null;
          const text = await downloadNotes(url.toString());
          const data = Uint8Array.from(text, (char) => char.charCodeAt(0));
          return { data, mediaType: 'text/plain' };
        })
      ),
  });
  return result.text;
}

export async function generateObjectWorkflow(): Promise<string> {
  const { output } = await generateText({
    model: temporalProvider.languageModel('gpt-4o-mini'),
    output: Output.object({
      schema: z.object({
        recipe: z.object({
          name: z.string(),
          ingredients: z.array(z.object({ name: z.string(), amount: z.string() })),
          steps: z.array(z.string()),
        }),
      }),
    }),
    prompt: 'Generate a lasagna recipe.',
  });
  return output.recipe.name;
}

export async function middlewareWorkflow(prompt: string): Promise<string> {
  const cache = new Map<string, any>();
  const middleware: LanguageModelMiddleware = {
    specificationVersion: 'v4',
    wrapGenerate: async ({ doGenerate, params }) => {
      const cacheKey = JSON.stringify(params);
      if (cache.has(cacheKey)) {
        return cache.get(cacheKey);
      }

      const result = await doGenerate();

      cache.set(cacheKey, result);

      return result;
    },
  };

  const model = wrapLanguageModel({
    model: temporalProvider.languageModel('gpt-4o-mini'),
    middleware,
  });

  const result = await generateText({
    model,
    prompt,
    instructions: 'You only respond in haikus.',
  });
  return result.text;
}

export async function telemetryWorkflow(prompt: string): Promise<string> {
  const result = await generateText({
    model: temporalProvider.languageModel('gpt-4o-mini'),
    prompt,
    instructions: 'You only respond in haikus.',
    telemetry: {
      isEnabled: true,
      integrations: new OpenTelemetry(),
    },
  });
  return result.text;
}

export async function mcpWorkflow(prompt: string): Promise<string> {
  const mcpClient = new TemporalMCPClient({ name: 'testServer' });
  const tools = await mcpClient.tools();
  const result = await generateText({
    model: temporalProvider.languageModel('gpt-4o-mini'),
    prompt,
    tools,
    instructions: 'What files do you have?',
    stopWhen: isStepCount(5),
  });
  return result.text;
}

/**
 * Test workflow that returns MCP tool schema structure for assertion.
 * Used to verify inputSchema survives activity serialization correctly.
 */
export async function mcpSchemaTestWorkflow(): Promise<{
  toolName: string;
  schemaType: string | undefined;
  hasProperties: boolean;
  propertyDescription: string | undefined;
}> {
  const mcpClient = new TemporalMCPClient({ name: 'testServer' });
  const tools = await mcpClient.tools();

  const [toolName, tool] = Object.entries(tools)[0]!;

  const schema = (tool as any).inputSchema.jsonSchema;

  return {
    toolName,
    schemaType: schema.type,
    hasProperties: schema.properties !== undefined,
    propertyDescription: schema.properties?.testParam?.description,
  };
}

/**
 * Workflow that demonstrates embedding model support.
 * Uses the temporalProvider to generate embeddings for multiple text values.
 */
export async function embeddingWorkflow(
  values: string[]
): Promise<{ count: number; dimensions: number; totalTokens?: number }> {
  const result = await embedMany({
    model: temporalProvider.embeddingModel('text-embedding-3-small'),
    values,
  });

  return {
    count: result.embeddings.length,
    dimensions: result.embeddings[0]?.length ?? 0,
    totalTokens: result.usage?.tokens,
  };
}

export async function chunkedEmbeddingWorkflow(
  values: string[],
  maxEmbeddingsPerCall?: number
): Promise<number[][]> {
  const provider = new TemporalProvider({ embeddingModel: { maxEmbeddingsPerCall } });
  const result = await embedMany({ model: provider.embeddingModel('echo'), values });
  return result.embeddings;
}

export async function streamingMetadataWorkflow(): Promise<LanguageModelV4StreamPart[]> {
  new WorkflowStream();
  const model = new TemporalLanguageModel('streaming-test-model', {
    streamingTopic: 'ai-sdk-test-stream',
    streamingBatchInterval: '1 millisecond',
  });
  const result = await model.doStream({
    prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
  });
  const parts: LanguageModelV4StreamPart[] = [];
  const reader = result.stream.getReader();

  while (true) {
    const { done, value } = await reader.read();
    if (done) return parts;
    parts.push(value);
  }
}

export async function slowModelWorkflow(
  call: 'stream' | 'generate' | 'embed',
  heartbeatTimeout?: number
): Promise<number> {
  new WorkflowStream();
  const activityOptions = { heartbeatTimeout, retry: { maximumAttempts: 1 } } as const;
  if (call === 'embed') {
    const model = new TemporalEmbeddingModel('slow-model', activityOptions);
    const result = await model.doEmbed({ values: ['test'] });
    return result.embeddings.length;
  }
  const model = new TemporalLanguageModel('slow-model', {
    ...activityOptions,
    streamingTopic: 'ai-sdk-test-stream',
    streamingBatchInterval: '1 millisecond',
  });
  const prompt = [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'test' }] }];
  if (call === 'generate') {
    const result = await model.doGenerate({ prompt });
    return result.content.length;
  }
  const result = await model.doStream({ prompt });
  const reader = result.stream.getReader();
  let count = 0;
  while (!(await reader.read()).done) count++;
  return count;
}

export const abortRequest = defineSignal('abortRequest');

function describeError(err: unknown): string {
  return `${(err as Error).name}|${err instanceof Error}`;
}

// A short heartbeat timeout lets the Activity learn about cancellation quickly.
const abortTestProvider = new TemporalProvider({
  languageModel: { heartbeatTimeout: '2 seconds' },
  embeddingModel: { heartbeatTimeout: '2 seconds' },
});

export async function abortableGenerateWorkflow(abortFirst: boolean): Promise<string> {
  const controller = new AbortController();
  if (abortFirst) controller.abort();
  setHandler(abortRequest, () => controller.abort());
  try {
    const result = await generateText({
      model: abortTestProvider.languageModel('gpt-4o-mini'),
      prompt: 'hi',
      abortSignal: controller.signal,
    });
    return `text:${result.text}`;
  } catch (err) {
    return `error:${describeError(err)}`;
  }
}

export async function abortableEmbedWorkflow(): Promise<number> {
  const result = await embedMany({
    model: abortTestProvider.embeddingModel('text-embedding-3-small'),
    values: ['a', 'b'],
    abortSignal: new AbortController().signal,
  });
  return result.embeddings.length;
}

export async function abortDuringEmbedWorkflow(): Promise<string> {
  const controller = new AbortController();
  setHandler(abortRequest, () => controller.abort());
  try {
    const result = await embedMany({
      model: abortTestProvider.embeddingModel('text-embedding-3-small'),
      values: ['a', 'b'],
      abortSignal: controller.signal,
    });
    return `count:${result.embeddings.length}`;
  } catch (err) {
    return `error:${describeError(err)}`;
  }
}

export async function abortableStreamWorkflow(): Promise<string[]> {
  new WorkflowStream();
  const controller = new AbortController();
  setHandler(abortRequest, () => controller.abort());
  const result = streamText({
    model: new TemporalLanguageModel('streaming-test-model', {
      streamingTopic: 'ai-sdk-abort-stream',
      heartbeatTimeout: '2 seconds',
    }),
    prompt: 'hi',
    abortSignal: controller.signal,
  });
  const types: string[] = [];
  for await (const part of result.stream) types.push(part.type);
  return types;
}
