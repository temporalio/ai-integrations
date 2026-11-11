/**
 * Test AI SDK integration with Temporal workflows
 */
import { randomUUID } from 'crypto';
import { readFileSync } from 'node:fs';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import path from 'node:path';
import type {
  EmbeddingModelV4,
  EmbeddingModelV4CallOptions,
  EmbeddingModelV4Result,
  ImageModelV4,
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4Content,
  LanguageModelV4FinishReason,
  LanguageModelV4GenerateResult,
  LanguageModelV4StreamResult,
  LanguageModelV4StreamPart,
  LanguageModelV4Usage,
  ProviderV4,
} from '@ai-sdk/provider';
import { APICallError } from '@ai-sdk/provider';
import { openai } from '@ai-sdk/openai';
import type { ExecutionContext, TestFn } from 'ava';
import { context, propagation, trace } from '@opentelemetry/api';
import { AsyncLocalStorageContextManager } from '@opentelemetry/context-async-hooks';
import { Resource } from '@opentelemetry/resources';
import {
  BasicTracerProvider,
  SimpleSpanProcessor,
  type ReadableSpan,
  type SpanExporter,
} from '@opentelemetry/sdk-trace-base';
import { SEMRESATTRS_SERVICE_NAME } from '@opentelemetry/semantic-conventions';
import { ExportResultCode, W3CTraceContextPropagator } from '@opentelemetry/core';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { ListToolsRequestSchema, CallToolRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { createMCPClient } from '@ai-sdk/mcp';
import { jsonSchema } from 'ai';
import { temporal } from '@temporalio/proto';
import { WorkflowClient, WorkflowFailedError } from '@temporalio/client';
import { ActivityFailure, ApplicationFailure, TemporalFailure } from '@temporalio/common';
import { optionalTsToMs } from '@temporalio/common/lib/time';
import type { OpenTelemetrySinks } from '@temporalio/interceptors-opentelemetry';
import {
  makeWorkflowExporter,
  OpenTelemetryActivityInboundInterceptor,
  OpenTelemetryActivityOutboundInterceptor,
  OpenTelemetryWorkflowClientCallsInterceptor,
  OpenTelemetryWorkflowClientInterceptor,
} from '@temporalio/interceptors-opentelemetry';
import { MockActivityEnvironment, workflowInterceptorModules } from '@temporalio/testing';
import { SimplePlugin } from '@temporalio/plugin';
import { bundleWorkflowCode, DefaultLogger, NativeConnection, Worker as CoreWorker } from '@temporalio/worker';
import type { InjectedSinks, WorkerOptions } from '@temporalio/worker';
import type { BaseContext } from '@temporalio/test-helpers';
import {
  test as anyTest,
  assertEventually,
  createBaseBundlerOptions,
  helpers,
  Worker,
  TestWorkflowEnvironment,
} from '@temporalio/test-helpers';
import { AiSdkPlugin, createActivities } from '..';
import type { InvokeModelArgs, InvokeModelResult } from '..';
import { startLocalWithRetry } from './helpers/environment';
import {
  binaryFileWorkflow,
  embeddingWorkflow,
  fileUrlWorkflow,
  generateObjectWorkflow,
  helloWorldAgent,
  imageToolWorkflow,
  mcpSchemaTestWorkflow,
  mcpWorkflow,
  middlewareWorkflow,
  slowModelWorkflow,
  streamingMetadataWorkflow,
  telemetryWorkflow,
  toolsWorkflow,
} from './workflows/ai-sdk';
import { getWeather } from './activities/ai-sdk';
import EventType = temporal.api.enums.v1.EventType;

const remoteTests = ['1', 't', 'true'].includes((process.env.AI_SDK_REMOTE_TESTS ?? 'false').toLowerCase());

export type ModelResponse = LanguageModelV4GenerateResult;

export class TestModel implements LanguageModelV4 {
  readonly specificationVersion = 'v4';
  readonly provider = 'temporal';
  readonly modelId = 'TestModel';
  private generator: Generator<ModelResponse>;
  private done: boolean = false;

  constructor(generator: Generator<ModelResponse>) {
    this.generator = generator;
  }

  get supportedUrls(): Record<string, RegExp[]> {
    return {};
  }

  async doGenerate(_: LanguageModelV4CallOptions): Promise<LanguageModelV4GenerateResult> {
    if (this.done) {
      throw new Error('Called generate more times than responses given to the test generator');
    }

    const result = this.generator.next();
    this.done = result.done ?? false;
    return result.value;
  }

  doStream(_options: LanguageModelV4CallOptions): PromiseLike<LanguageModelV4StreamResult> {
    throw new Error('Streaming not supported.');
  }
}

export type EmbeddingResponse = EmbeddingModelV4Result;

export class TestEmbeddingModel implements EmbeddingModelV4 {
  readonly specificationVersion = 'v4';
  readonly provider = 'temporal';
  readonly modelId = 'TestEmbeddingModel';
  readonly maxEmbeddingsPerCall = undefined;
  readonly supportsParallelCalls = true;
  private generator: Generator<EmbeddingResponse>;
  private done: boolean = false;

  constructor(generator: Generator<EmbeddingResponse>) {
    this.generator = generator;
  }

  async doEmbed(_options: EmbeddingModelV4CallOptions): Promise<EmbeddingModelV4Result> {
    if (this.done) {
      throw new Error('Called embed more times than responses given to the test generator');
    }

    const result = this.generator.next();
    this.done = result.done ?? false;
    return result.value;
  }
}

export class TestProvider implements ProviderV4 {
  readonly specificationVersion = 'v4';
  private languageModelGenerator: Generator<ModelResponse>;
  private embeddingModelGenerator?: Generator<EmbeddingResponse>;

  constructor(
    languageModelGenerator: Generator<ModelResponse>,
    embeddingModelGenerator?: Generator<EmbeddingResponse>
  ) {
    this.languageModelGenerator = languageModelGenerator;
    this.embeddingModelGenerator = embeddingModelGenerator;
  }

  imageModel(_modelId: string): ImageModelV4 {
    throw new Error('Not implemented');
  }

  languageModel(_modelId: string): LanguageModelV4 {
    return new TestModel(this.languageModelGenerator);
  }

  embeddingModel(_modelId: string): EmbeddingModelV4 {
    if (!this.embeddingModelGenerator) {
      throw new Error('Embedding model generator not provided');
    }
    return new TestEmbeddingModel(this.embeddingModelGenerator);
  }
}

class StreamingTestProvider implements ProviderV4 {
  readonly specificationVersion = 'v4';

  constructor(private readonly parts: LanguageModelV4StreamPart[]) {}

  languageModel(_modelId: string): LanguageModelV4 {
    const parts = this.parts;
    return {
      specificationVersion: 'v4',
      provider: 'temporal-test',
      modelId: 'streaming-test-model',
      supportedUrls: {},
      async doGenerate(): Promise<LanguageModelV4GenerateResult> {
        throw new Error('Generate not supported.');
      },
      async doStream(): Promise<LanguageModelV4StreamResult> {
        return {
          stream: new ReadableStream({
            start(controller) {
              for (const part of parts) controller.enqueue(part);
              controller.close();
            },
          }),
        };
      },
    };
  }

  embeddingModel(_modelId: string): EmbeddingModelV4 {
    throw new Error('Not implemented.');
  }

  imageModel(_modelId: string): ImageModelV4 {
    throw new Error('Not implemented.');
  }
}

function createFinishReason(
  unified: 'stop' | 'length' | 'content-filter' | 'tool-calls' | 'error' | 'other'
): LanguageModelV4FinishReason {
  return { unified, raw: undefined };
}

function createUsage(inputTokens: number = 10, outputTokens: number = 20): LanguageModelV4Usage {
  return {
    inputTokens: {
      total: inputTokens,
      noCache: undefined,
      cacheRead: undefined,
      cacheWrite: undefined,
    },
    outputTokens: {
      total: outputTokens,
      text: undefined,
      reasoning: undefined,
    },
  };
}

function contentResponse(
  content: LanguageModelV4Content[],
  finishReason: LanguageModelV4FinishReason = createFinishReason('stop')
): ModelResponse {
  return {
    content,
    finishReason,
    usage: createUsage(),
    warnings: [],
  };
}

function textResponse(content: string): ModelResponse {
  return contentResponse([
    {
      type: 'text',
      text: content,
    },
  ]);
}

function toolCallResponse(toolName: string, input: string): ModelResponse {
  return contentResponse(
    [
      {
        type: 'tool-call',
        toolCallId: 'call_yY3nlDwH5BQSJo63qC61L4ZB',
        toolName,
        input,
      },
    ],
    createFinishReason('tool-calls')
  );
}

function embeddingResponse(values: string[]): EmbeddingResponse {
  // Generate mock embeddings (1536-dimensional vectors like OpenAI's text-embedding-3-small)
  const embeddings = values.map((_, index) => {
    const embedding = new Array(1536).fill(0).map((_, i) => Math.sin(index + i * 0.01));
    return embedding;
  });
  return {
    embeddings,
    usage: {
      tokens: values.reduce((acc, v) => acc + v.length, 0),
    },
    warnings: [],
  };
}

function* helloWorkflowGenerator(): Generator<ModelResponse> {
  yield textResponse('Test Haiku');
}

function* embeddingGenerator(): Generator<EmbeddingResponse> {
  yield embeddingResponse(['Hello, world!', 'How are you?']);
}

const test = anyTest as TestFn<BaseContext>;

test.before(async (t) => {
  const env = await startLocalWithRetry(() => TestWorkflowEnvironment.createLocal());
  const workflowBundle = await bundleWorkflowCode({
    ...createBaseBundlerOptions(),
    workflowsPath: require.resolve('./workflows/ai-sdk'),
    workflowInterceptorModules,
    logger: new DefaultLogger('WARN'),
    // The plugin's configureBundler prepends the polyfill installer to the bundle entry;
    // the test workflows import `ai` before the workflow entry point and rely on it.
    plugins: [new AiSdkPlugin({ modelProvider: new TestProvider(helloWorkflowGenerator()) })],
  });
  t.context = { env, workflowBundle };
});

test.after.always(async (t) => {
  await t.context.env?.teardown();
});

test('Hello world agent responds in haikus', async (t) => {
  if (remoteTests) {
    t.timeout(120 * 1000);
  }
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new AiSdkPlugin({
        modelProvider: remoteTests ? openai : new TestProvider(helloWorkflowGenerator()),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(helloWorldAgent, {
      args: ['Tell me about recursion in programming.'],
    });

    t.assert(result);
    if (!remoteTests) {
      t.is('Test Haiku', result);
    }
  });
});

function* toolsWorkflowGenerator(): Generator<ModelResponse> {
  yield toolCallResponse('getWeather', '{"location":"Tokyo"}');
  yield textResponse('Test weather result');
}

test('Tools workflow can use AI tools', async (t) => {
  if (remoteTests) {
    t.timeout(120 * 1000);
  }
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new AiSdkPlugin({
        modelProvider: remoteTests ? openai : new TestProvider(toolsWorkflowGenerator()),
      }),
    ],
    activities: {
      getWeather,
    },
  });
  await worker.runUntil(async () => {
    const handle = await startWorkflow(toolsWorkflow, {
      args: ['What is the weather in Tokyo?'],
      workflowExecutionTimeout: '10 seconds',
    });

    const result = await handle.result();

    t.assert(result);
    if (!remoteTests) {
      t.is('Test weather result', result);

      // Check that activities were scheduled
      const { events } = await handle.fetchHistory();
      const activityCompletedEvents =
        events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED) ?? [];

      // Should have at least 2 events: invokeModel and getWeather
      t.assert(
        activityCompletedEvents.length >= 2,
        `Expected at least 2 activity completions, got ${activityCompletedEvents.length}`
      );

      // Check that getWeather activity was called
      const activityTypes = activityCompletedEvents.map(
        (e) => e?.activityTaskScheduledEventAttributes?.activityType?.name
      );
      t.assert(activityTypes.includes('getWeather'), 'getWeather activity should have been called');
    }
  });
});

function* imageToolWorkflowGenerator(): Generator<ModelResponse> {
  yield toolCallResponse('screenshot', '{}');
  yield textResponse('A transparent image');
}

test('Image tool results are converted inside the workflow sandbox', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);
  const worker = await createWorker({
    plugins: [new AiSdkPlugin({ modelProvider: new TestProvider(imageToolWorkflowGenerator()) })],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(imageToolWorkflow, {
      workflowExecutionTimeout: '10 seconds',
    });
    t.is(result, 'A transparent image');
  });
});

test('Binary file data reaches the model and the workflow as the same bytes', async (t) => {
  const promptBytes = [137, 80, 78, 71, 0, 255];
  const generatedBytes = [0, 1, 2, 254, 255];
  const received: LanguageModelV4CallOptions[] = [];
  function* generator(): Generator<ModelResponse> {
    yield contentResponse([
      {
        type: 'file',
        mediaType: 'application/octet-stream',
        data: { type: 'data', data: new Uint8Array(generatedBytes) },
      },
    ]);
  }
  const provider = new TestProvider(generator());
  const languageModel = provider.languageModel.bind(provider);
  provider.languageModel = (modelId) => {
    const model = languageModel(modelId);
    const doGenerate = model.doGenerate.bind(model);
    model.doGenerate = (options) => {
      received.push(options);
      return doGenerate(options);
    };
    return model;
  };
  const { createWorker, executeWorkflow } = helpers(t);
  const worker = await createWorker({ plugins: [new AiSdkPlugin({ modelProvider: provider })] });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(binaryFileWorkflow, {
      args: [promptBytes],
      workflowExecutionTimeout: '10 seconds',
    });
    t.deepEqual(result, generatedBytes);
  });

  const userParts = received[0]!.prompt.flatMap((m) => (m.role === 'user' ? m.content : []));
  const file = userParts.find((p) => p.type === 'file');
  const data = file?.data.type === 'data' ? file.data.data : undefined;
  t.is(typeof data, 'string');
  t.deepEqual([...Buffer.from(data as string, 'base64')], promptBytes);
class RecordingProvider implements ProviderV4 {
  readonly specificationVersion = 'v4';
  readonly calls: LanguageModelV4CallOptions[] = [];

  constructor(private readonly supportedUrls: Record<string, RegExp[]>) {}

  languageModel(_modelId: string): LanguageModelV4 {
    const { calls, supportedUrls } = this;
    return {
      specificationVersion: 'v4',
      provider: 'temporal-test',
      modelId: 'recording-test-model',
      supportedUrls,
      async doGenerate(options): Promise<LanguageModelV4GenerateResult> {
        calls.push(options);
        return textResponse('seen');
      },
      async doStream(): Promise<LanguageModelV4StreamResult> {
        throw new Error('Streaming not supported.');
      },
    };
  }

  embeddingModel(_modelId: string): EmbeddingModelV4 {
    throw new Error('Not implemented.');
  }

  imageModel(_modelId: string): ImageModelV4 {
    throw new Error('Not implemented.');
  }
}

const pngBytes = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5E' +
    'rkJggg==',
  'base64'
);

async function serveFiles(t: { teardown(fn: () => Promise<void>): void }) {
  const requested: string[] = [];
  const server = http.createServer((req, res) => {
    requested.push(req.url ?? '');
    if (req.url === '/image') {
      res.writeHead(200, { 'content-type': 'image/png' }).end(pngBytes);
    } else if (req.url === '/notes') {
      res.writeHead(200, { 'content-type': 'text/plain; charset=utf-8' }).end('some notes');
    } else {
      res.writeHead(404).end();
    }
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  t.teardown(() => new Promise<void>((resolve) => server.close(() => resolve())));
  const baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  return { baseUrl, requested };
}

// The default `createDownload()` refuses private addresses such as 127.0.0.1.
async function localDownload({ url, abortSignal }: { url: URL; abortSignal?: AbortSignal }) {
  const response = await fetch(url, { signal: abortSignal });
  const data = new Uint8Array(await response.arrayBuffer());
  return { data, mediaType: response.headers.get('content-type') ?? undefined };
}

async function runFileUrlWorkflow(t: ExecutionContext<BaseContext>, provider: RecordingProvider) {
  const server = await serveFiles(t);
  const { createWorker, executeWorkflow } = helpers(t);
  const worker = await createWorker({
    plugins: [new AiSdkPlugin({ modelProvider: provider, download: localDownload })],
  });
  await worker.runUntil(async () => {
    const result = await executeWorkflow(fileUrlWorkflow, {
      args: [server.baseUrl],
      workflowExecutionTimeout: '10 seconds',
    });
    t.is(result, 'seen');
  });
  t.is(provider.calls.length, 1);
  const [message] = provider.calls[0]!.prompt;
  if (message?.role !== 'user') throw new Error('Expected a user message');
  return { ...server, content: message.content };
}

test('Prompt file URLs the model does not support are downloaded in the activity', async (t) => {
  const { content, requested } = await runFileUrlWorkflow(t, new RecordingProvider({}));

  t.deepEqual(content.slice(1), [
    {
      type: 'file',
      mediaType: 'image/png',
      data: { type: 'data', data: new Uint8Array(pngBytes) },
    },
    {
      type: 'file',
      mediaType: 'text/plain',
      data: { type: 'data', data: new TextEncoder().encode('some notes') },
    },
  ]);
  t.deepEqual(requested.sort(), ['/image', '/notes']);
});

test('Prompt file URLs the model supports reach the model as URL objects', async (t) => {
  const provider = new RecordingProvider({ 'image/*': [/^http:\/\/127\.0\.0\.1:/] });
  const { content, requested, baseUrl } = await runFileUrlWorkflow(t, provider);

  const [, image, notes] = content;
  if (image?.type !== 'file' || image.data.type !== 'url') throw new Error('Expected a URL part');
  t.true(image.data.url instanceof URL);
  t.is(image.data.url.href, `${baseUrl}/image`);
  t.is(image.mediaType, 'image');
  t.like(notes, { mediaType: 'text/plain', data: { type: 'data' } });
  t.deepEqual(requested, ['/notes']);
});

test('Streaming text, finish and response metadata survive activity execution and workflow replay', async (t) => {
  const providerMetadata = { test: { source: 'delta' } };
  const finishMetadata = { test: { cachedTokens: 7 } };
  const responseMetadata = { id: 'response-1', modelId: 'test-model', timestamp: new Date('2026-01-01T00:00:00Z') };
  const provider = new StreamingTestProvider([
    { type: 'stream-start', warnings: [] },
    { type: 'response-metadata', ...responseMetadata },
    { type: 'text-start', id: 'provider-part' },
    { type: 'text-delta', id: 'provider-part', delta: 'hello', providerMetadata },
    { type: 'text-end', id: 'provider-part' },
    { type: 'finish', finishReason: createFinishReason('stop'), usage: createUsage(), providerMetadata: finishMetadata },
  ]);
  const { createWorker, executeWorkflow } = helpers(t);
  const worker = await createWorker({ plugins: [new AiSdkPlugin({ modelProvider: provider })] });

  await worker.runUntil(async () => {
    const parts = await executeWorkflow(streamingMetadataWorkflow, {
      workflowExecutionTimeout: '10 seconds',
    });

    t.deepEqual(parts, [
      { type: 'stream-start', warnings: [] },
      { type: 'response-metadata', ...responseMetadata, timestamp: responseMetadata.timestamp.toISOString() },
      { type: 'text-start', id: 'part-0', providerMetadata },
      { type: 'text-delta', id: 'part-0', delta: 'hello' },
      { type: 'text-end', id: 'part-0' },
      {
        type: 'finish',
        providerMetadata: finishMetadata,
        finishReason: { unified: 'stop' },
        usage: { inputTokens: { total: 10 }, outputTokens: { total: 20 } },
      },
    ]);
  });
});

// Each call waits longer than the 1-second `heartbeatTimeout` the test below sets first.
const slowModelDelayMs = 2000;

class SlowTestProvider implements ProviderV4 {
  readonly specificationVersion = 'v4';

  languageModel(_modelId: string): LanguageModelV4 {
    return {
      specificationVersion: 'v4',
      provider: 'temporal-test',
      modelId: 'slow-model',
      supportedUrls: {},
      async doGenerate(): Promise<LanguageModelV4GenerateResult> {
        await new Promise((resolve) => setTimeout(resolve, slowModelDelayMs));
        return textResponse('slow');
      },
      async doStream(): Promise<LanguageModelV4StreamResult> {
        return {
          stream: new ReadableStream({
            async start(controller) {
              await new Promise((resolve) => setTimeout(resolve, slowModelDelayMs));
              controller.enqueue({ type: 'stream-start', warnings: [] });
              controller.enqueue({
                type: 'finish',
                finishReason: createFinishReason('stop'),
                usage: createUsage(),
              });
              controller.close();
            },
          }),
        };
      },
    };
  }

  embeddingModel(_modelId: string): EmbeddingModelV4 {
    return {
      specificationVersion: 'v4',
      provider: 'temporal-test',
      modelId: 'slow-model',
      maxEmbeddingsPerCall: undefined,
      supportsParallelCalls: true,
      async doEmbed(): Promise<EmbeddingModelV4Result> {
        await new Promise((resolve) => setTimeout(resolve, slowModelDelayMs));
        return embeddingResponse(['test']);
      },
    };
  }

  imageModel(_modelId: string): ImageModelV4 {
    throw new Error('Not implemented.');
  }
}

for (const [call, expected] of [
  ['stream', 2],
  ['generate', 1],
  ['embed', 1],
] as const) {
  test(`Model activity (${call}) heartbeats while the provider is slow`, async (t) => {
    const { createWorker, startWorkflow } = helpers(t);
    const plugin = new AiSdkPlugin({ modelProvider: new SlowTestProvider() });
    const worker = await createWorker({ plugins: [plugin] });

    await worker.runUntil(async () => {
      // A timeout you set wins, none gives the 10-second default, and 0 turns it off.
      for (const [heartbeatTimeout, expectedMs] of [
        [1000, 1000],
        [undefined, 10_000],
        [0, undefined],
      ] as const) {
        const handle = await startWorkflow(slowModelWorkflow, {
          args: [call, heartbeatTimeout],
          workflowExecutionTimeout: '10 seconds',
        });
        t.is(await handle.result(), expected);
        const { events } = await handle.fetchHistory();
        const heartbeatTimeouts = events
          ?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED)
          .map((e) => optionalTsToMs(e.activityTaskScheduledEventAttributes?.heartbeatTimeout) || undefined);
        t.deepEqual(heartbeatTimeouts, [expectedMs]);
      }
    });
  });
}

function* generateObjectWorkflowGenerator(): Generator<ModelResponse> {
  yield textResponse(
    '{"recipe":{"name":"Classic Lasagna","ingredients":[],"steps":["Preheat the oven to 375°F (190°C)."]}}'
  );
}

test('Generate object', async (t) => {
  if (remoteTests) {
    t.timeout(120 * 1000);
  }
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new AiSdkPlugin({
        modelProvider: remoteTests ? openai : new TestProvider(generateObjectWorkflowGenerator()),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(generateObjectWorkflow, {
      args: ['Tell me about recursion in programming.'],
      workflowExecutionTimeout: '30 seconds',
    });

    t.assert(result);
    if (!remoteTests) {
      t.is('Classic Lasagna', result);
    }
  });
});

test('Middleware', async (t) => {
  if (remoteTests) {
    t.timeout(120 * 1000);
  }
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new AiSdkPlugin({
        modelProvider: remoteTests ? openai : new TestProvider(helloWorkflowGenerator()),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(middlewareWorkflow, {
      args: ['Tell me about recursion in programming.'],
      workflowExecutionTimeout: '30 seconds',
    });

    t.assert(result);
    if (!remoteTests) {
      t.is('Test Haiku', result);
    }
  });
});

test('Embedding model generates embeddings', async (t) => {
  if (remoteTests) {
    t.timeout(120 * 1000);
  }
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new AiSdkPlugin({
        modelProvider: remoteTests ? openai : new TestProvider(helloWorkflowGenerator(), embeddingGenerator()),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(embeddingWorkflow, {
      args: [['Hello, world!', 'How are you?']],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();

    t.assert(result);
    t.is(result.count, 2);
    t.is(result.dimensions, 1536);

    // Check that the embedding activity was scheduled
    const { events } = await handle.fetchHistory();
    const activityScheduledEvents =
      events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED) ?? [];

    // Should have at least 1 event: invokeEmbeddingModel
    t.assert(
      activityScheduledEvents.length >= 1,
      `Expected at least 1 activity scheduled, got ${activityScheduledEvents.length}`
    );

    // Check that invokeEmbeddingModel activity was called
    const activityTypes = activityScheduledEvents.map(
      (e) => e?.activityTaskScheduledEventAttributes?.activityType?.name
    );
    t.assert(activityTypes.includes('invokeEmbeddingModel'), 'invokeEmbeddingModel activity should have been called');
  });
});

function apiCallError(statusCode: number, isRetryable: boolean): APICallError {
  return new APICallError({
    message: `Provider returned ${statusCode}`,
    url: 'https://provider.invalid/v1/chat',
    requestBodyValues: {},
    statusCode,
    isRetryable,
  });
}

/** Throws the scripted errors in order, then answers with `response`. */
class FailingProvider implements ProviderV4 {
  readonly specificationVersion = 'v4';
  calls = 0;

  constructor(
    private readonly errors: unknown[],
    private readonly response: ModelResponse = textResponse('Recovered')
  ) {}

  private next(): void {
    const error = this.errors[this.calls++];
    if (error !== undefined) throw error;
  }

  languageModel(_modelId: string): LanguageModelV4 {
    return {
      specificationVersion: 'v4',
      provider: 'temporal-test',
      modelId: 'failing-test-model',
      supportedUrls: {},
      doGenerate: async (): Promise<LanguageModelV4GenerateResult> => {
        this.next();
        return this.response;
      },
      doStream: async (): Promise<LanguageModelV4StreamResult> => {
        throw new Error('Streaming not supported.');
      },
    };
  }

  embeddingModel(_modelId: string): EmbeddingModelV4 {
    return {
      specificationVersion: 'v4',
      provider: 'temporal-test',
      modelId: 'failing-test-embedding-model',
      maxEmbeddingsPerCall: undefined,
      supportsParallelCalls: true,
      doEmbed: async (): Promise<EmbeddingModelV4Result> => {
        this.next();
        return embeddingResponse(['unused']);
      },
    };
  }

  imageModel(_modelId: string): ImageModelV4 {
    throw new Error('Not implemented.');
  }
}

test('A non-retryable provider error fails the Workflow after one Activity attempt', async (t) => {
  const provider = new FailingProvider([apiCallError(401, false)]);
  const { createWorker, startWorkflow } = helpers(t);
  const worker = await createWorker({ plugins: [new AiSdkPlugin({ modelProvider: provider })] });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(helloWorldAgent, {
      args: ['hi'],
      workflowExecutionTimeout: '30 seconds',
    });
    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    const activityFailure = err!.cause;
    t.true(activityFailure instanceof ActivityFailure);
    const failure = activityFailure!.cause;
    if (!(failure instanceof ApplicationFailure)) {
      t.fail(`Expected an ApplicationFailure, got ${failure}`);
      return;
    }
    t.true(failure.nonRetryable);
    t.is(failure.type, 'APICallError');
    t.is(failure.message, 'Provider returned 401');
    t.true(failure.cause instanceof TemporalFailure);
    t.is(failure.cause?.message, 'Provider returned 401');

    const { events } = await handle.fetchHistory();
    const failed = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED);
    t.is(failed?.length, 1);
  });
  t.is(provider.calls, 1);
});

test('A retryable provider error is retried by the Activity retry policy', async (t) => {
  const provider = new FailingProvider([apiCallError(503, true)]);
  const { createWorker, executeWorkflow } = helpers(t);
  const worker = await createWorker({ plugins: [new AiSdkPlugin({ modelProvider: provider })] });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(helloWorldAgent, {
      args: ['hi'],
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(result, 'Recovered');
  });
  t.is(provider.calls, 2);
});

test('invokeEmbeddingModel fails without retry on a non-retryable provider error', async (t) => {
  const activities = createActivities(new FailingProvider([apiCallError(400, false)])) as {
    invokeEmbeddingModel: (args: { modelId: string; values: string[] }) => Promise<unknown>;
  };
  const env = new MockActivityEnvironment();
  const err = await t.throwsAsync(
    env.run(activities.invokeEmbeddingModel, { modelId: 'm', values: ['a'] }),
    { instanceOf: ApplicationFailure }
  );
  t.true(err!.nonRetryable);
  t.is(err!.type, 'APICallError');
  t.true(APICallError.isInstance(err!.cause));
});

test.serial('Telemetry', async (t) => {
  if (remoteTests) {
    t.timeout(120 * 1000);
  }
  let provider: BasicTracerProvider | undefined;
  try {
    const spans = Array<ReadableSpan>();

    const staticResource = new Resource({
      [SEMRESATTRS_SERVICE_NAME]: 'ts-test-otel-worker',
    });
    const traceExporter: SpanExporter = {
      export(spans_, resultCallback) {
        spans.push(...spans_);
        resultCallback({ code: ExportResultCode.SUCCESS });
      },
      async shutdown() {
        // Nothing to shutdown
      },
    };
    provider = new BasicTracerProvider({
      resource: staticResource,
      spanProcessors: [new SimpleSpanProcessor(traceExporter)],
    });
    provider.register({
      contextManager: new AsyncLocalStorageContextManager().enable(),
      propagator: new W3CTraceContextPropagator(),
    });
    const sinks: InjectedSinks<OpenTelemetrySinks> = {
      exporter: makeWorkflowExporter(traceExporter, staticResource), // eslint-disable-line @typescript-eslint/no-deprecated
    };

    const worker = await Worker.create({
      connection: t.context.env.nativeConnection,
      namespace: t.context.env.namespace,
      plugins: [
        new AiSdkPlugin({
          modelProvider: remoteTests ? openai : new TestProvider(helloWorkflowGenerator()),
        }),
      ],
      taskQueue: 'test-ai-telemetry',
      workflowsPath: require.resolve('./workflows/ai-sdk'),
      bundlerOptions: createBaseBundlerOptions(),

      interceptors: {
        client: {
          workflow: [new OpenTelemetryWorkflowClientCallsInterceptor()],
        },
        workflowModules: [require.resolve('./workflows/otel-interceptors')],
        activity: [
          (ctx) => ({
            inbound: new OpenTelemetryActivityInboundInterceptor(ctx),
            outbound: new OpenTelemetryActivityOutboundInterceptor(ctx),
          }),
        ],
      },
      sinks,
    });

    const client = new WorkflowClient({
      connection: t.context.env.connection,
      namespace: t.context.env.namespace,
      interceptors: [new OpenTelemetryWorkflowClientInterceptor()],
    });
    await worker.runUntil(async () => {
      await client.execute(telemetryWorkflow, {
        taskQueue: 'test-ai-telemetry',
        workflowId: randomUUID(),
        args: ['Tell me about recursion'],
      });
    });
    t.log(
      'exported spans:',
      spans.map(({ name }) => name)
    );
    // @ai-sdk/otel's OpenTelemetry integration names spans `${operationName} ${modelId}`
    // following the GenAI semantic conventions; generateText maps to `invoke_agent`.
    const generateSpan = spans.find(({ name }) => name === `invoke_agent gpt-4o-mini`);
    t.true(generateSpan !== undefined);
  } finally {
    await provider?.shutdown();
    context.disable();
    propagation.disable();
    trace.disable();
  }
});

function* mcpGenerator(): Generator<ModelResponse> {
  yield toolCallResponse('list_directory', '/');
  yield textResponse('Some files');
}

test('callToolActivity awaits tool.execute before closing MCP client', async (t) => {
  // Track the order of operations to verify close() is called AFTER execute() completes
  const operationOrder: string[] = [];

  // Create a mock MCP client that tracks when close() is called
  // We only need to implement the methods used by callToolActivity: tools() and close()
  const mockMcpClient = {
    async tools() {
      return {
        testTool: {
          description: 'A test tool',
          inputSchema: { type: 'object' },
          execute: async (_input: unknown, _options: unknown) => {
            operationOrder.push('execute-start');
            // Simulate async work - without await in the activity, close() will be called before this resolves
            await new Promise((resolve) => setTimeout(resolve, 50));
            operationOrder.push('execute-end');
            return { result: 'success' };
          },
        },
      };
    },
    async close() {
      operationOrder.push('close');
    },
  };

  // Create a factory that returns the mock client - cast to any since we only implement the methods we need

  const mockMcpClientFactory = async () => mockMcpClient as any;

  // Create activities with the mock MCP client factory. mcpConnectionIdleTimeout: 0 opts out of
  // connection reuse so this test's create-then-close-per-call assertion still holds.
  const activities = createActivities(
    new TestProvider(helloWorkflowGenerator()),
    { testServer: mockMcpClientFactory },
    { mcpConnectionIdleTimeout: 0 }
  ) as Record<string, (args: unknown) => Promise<unknown>>;

  // Get the callTool activity
  const callToolActivity = activities['testServer-callTool']!;
  t.truthy(callToolActivity, 'callToolActivity should exist');

  // Call the activity
  const result = await callToolActivity({
    name: 'testTool',
    input: {},
    options: {},
  });

  // Verify the result
  t.deepEqual(result, { result: 'success' });

  // Verify the order: execute should complete BEFORE close is called
  t.deepEqual(
    operationOrder,
    ['execute-start', 'execute-end', 'close'],
    'close() should be called after execute() completes, not before'
  );
});

test('MCP client connection is reused across repeated tools()/callTool invocations within the idle window', async (t) => {
  let factoryCalls = 0;
  const closeCalls: string[] = [];

  const mockMcpClientFactory = async () => {
    factoryCalls++;
    const clientId = `client-${factoryCalls}`;
    return {
      async tools() {
        return {
          testTool: {
            description: 'A test tool',
            // The real @ai-sdk/mcp client always normalizes tool input schemas into a Schema
            // object via `jsonSchema()` before returning them from `tools()`; a bare JSON
            // Schema literal (e.g. `{ type: 'object' }`) is not a valid `FlexibleSchema` and
            // makes `asSchema()` in `extractTools` (used by `listToolsActivity`) throw.
            inputSchema: jsonSchema({ type: 'object' }),
            execute: async () => ({ result: clientId }),
          },
        };
      },
      async close() {
        closeCalls.push(clientId);
      },
    } as any;
  };

  const activities = createActivities(
    new TestProvider(helloWorkflowGenerator()),
    { reuseServer: mockMcpClientFactory },
    { mcpConnectionIdleTimeout: '1 minute' }
  ) as Record<string, (args: unknown) => Promise<unknown>>;

  const listToolsActivity = activities['reuseServer-listTools']!;
  const callToolActivity = activities['reuseServer-callTool']!;

  await listToolsActivity({});
  const firstResult = await callToolActivity({ name: 'testTool', input: {}, options: {} });
  const secondResult = await callToolActivity({ name: 'testTool', input: {}, options: {} });

  t.is(factoryCalls, 1, 'mcpClientFactory should only be called once while the connection is reused');
  t.deepEqual(firstResult, { result: 'client-1' });
  t.deepEqual(secondResult, { result: 'client-1' });
  t.deepEqual(closeCalls, [], 'the connection should not be closed while still within the idle window');
});

test('idle MCP client connection is closed and evicted after the idle timeout elapses', async (t) => {
  let factoryCalls = 0;
  const closeCalls: string[] = [];

  const mockMcpClientFactory = async () => {
    factoryCalls++;
    const clientId = `client-${factoryCalls}`;
    return {
      async tools() {
        return {
          testTool: {
            description: 'A test tool',
            inputSchema: { type: 'object' },
            execute: async () => ({ result: clientId }),
          },
        };
      },
      async close() {
        closeCalls.push(clientId);
      },
    } as any;
  };

  const activities = createActivities(
    new TestProvider(helloWorkflowGenerator()),
    { idleEvictServer: mockMcpClientFactory },
    { mcpConnectionIdleTimeout: 20 }
  ) as Record<string, (args: unknown) => Promise<unknown>>;

  const callToolActivity = activities['idleEvictServer-callTool']!;

  await callToolActivity({ name: 'testTool', input: {}, options: {} });
  t.is(factoryCalls, 1);
  t.deepEqual(closeCalls, [], 'connection should still be open immediately after the call');

  // Timer callbacks can be delayed on busy runners; wait for eviction instead of a fixed sleep.
  await assertEventually(t, (t) => {
    t.deepEqual(closeCalls, ['client-1'], 'connection should be closed once the idle timeout elapses');
  });

  // A subsequent call should reconnect via a fresh factory call.
  await callToolActivity({ name: 'testTool', input: {}, options: {} });
  t.is(factoryCalls, 2, 'a new connection should be created after eviction');
});

test('a long-running call is not evicted mid-call by a short idle timeout', async (t) => {
  let factoryCalls = 0;
  const closeCalls: string[] = [];

  const mockMcpClientFactory = async () => {
    factoryCalls++;
    const clientId = `client-${factoryCalls}`;
    return {
      async tools() {
        return {
          testTool: {
            description: 'A test tool',
            inputSchema: { type: 'object' },
            execute: async () => {
              // Run well past the configured idle timeout while the call is in flight.
              await new Promise((resolve) => setTimeout(resolve, 100));
              return { result: clientId };
            },
          },
        };
      },
      async close() {
        closeCalls.push(clientId);
      },
    } as any;
  };

  const activities = createActivities(
    new TestProvider(helloWorkflowGenerator()),
    { longCallServer: mockMcpClientFactory },
    { mcpConnectionIdleTimeout: 20 }
  ) as Record<string, (args: unknown) => Promise<unknown>>;

  const callToolActivity = activities['longCallServer-callTool']!;

  const result = await callToolActivity({ name: 'testTool', input: {}, options: {} });

  t.deepEqual(result, { result: 'client-1' });
  t.deepEqual(closeCalls, [], 'the connection must not be closed by the idle timer while the call is still in flight');
  t.is(factoryCalls, 1, 'the in-flight call should not have triggered a reconnect');
});

test('a tool-level error on one concurrent call does not evict the connection out from under a sibling call', async (t) => {
  let factoryCalls = 0;
  const closeCalls: string[] = [];
  let markHealthyCallStarted!: () => void;
  const healthyCallStarted = new Promise<void>((resolve) => {
    markHealthyCallStarted = resolve;
  });
  let releaseHealthyCall!: () => void;
  const healthyCallCanFinish = new Promise<void>((resolve) => {
    releaseHealthyCall = resolve;
  });

  const mockMcpClientFactory = async () => {
    factoryCalls++;
    const clientId = `client-${factoryCalls}`;
    return {
      async tools() {
        return {
          failingTool: {
            description: 'A tool that always fails',
            inputSchema: { type: 'object' },
            execute: async () => {
              await healthyCallStarted;
              throw new Error('business logic failure, unrelated to the connection');
            },
          },
          slowTool: {
            description: 'A slow but healthy tool',
            inputSchema: { type: 'object' },
            execute: async () => {
              markHealthyCallStarted();
              await healthyCallCanFinish;
              return { result: clientId };
            },
          },
        };
      },
      async close() {
        closeCalls.push(clientId);
      },
    } as any;
  };

  const activities = createActivities(
    new TestProvider(helloWorkflowGenerator()),
    { sharedServer: mockMcpClientFactory },
    { mcpConnectionIdleTimeout: '1 minute' }
  ) as Record<string, (args: unknown) => Promise<unknown>>;

  const callToolActivity = activities['sharedServer-callTool']!;

  const failingCall = callToolActivity({ name: 'failingTool', input: {}, options: {} });
  const healthyCall = callToolActivity({ name: 'slowTool', input: {}, options: {} });

  await t.throwsAsync(failingCall, { message: /business logic failure/ });
  const closeCallsWhileSiblingInFlight = [...closeCalls];
  releaseHealthyCall();

  const healthyResult = await healthyCall;
  t.deepEqual(
    closeCallsWhileSiblingInFlight,
    [],
    'the shared connection must not be closed while the sibling call is still in flight'
  );
  t.deepEqual(healthyResult, { result: 'client-1' }, 'the sibling call should complete successfully');
  t.is(factoryCalls, 1, 'both concurrent calls should have shared the same pooled connection');
});

test('mcpConnectionIdleTimeout: 0 opts out of connection reuse (create-then-close every call)', async (t) => {
  let factoryCalls = 0;
  const closeCalls: string[] = [];

  const mockMcpClientFactory = async () => {
    factoryCalls++;
    const clientId = `client-${factoryCalls}`;
    return {
      async tools() {
        return {
          testTool: {
            description: 'A test tool',
            inputSchema: { type: 'object' },
            execute: async () => ({ result: clientId }),
          },
        };
      },
      async close() {
        closeCalls.push(clientId);
      },
    } as any;
  };

  const activities = createActivities(
    new TestProvider(helloWorkflowGenerator()),
    { optOutServer: mockMcpClientFactory },
    { mcpConnectionIdleTimeout: 0 }
  ) as Record<string, (args: unknown) => Promise<unknown>>;

  const callToolActivity = activities['optOutServer-callTool']!;

  await callToolActivity({ name: 'testTool', input: {}, options: {} });
  await callToolActivity({ name: 'testTool', input: {}, options: {} });

  t.is(factoryCalls, 2, 'a fresh client should be created for every call when reuse is disabled');
  t.deepEqual(closeCalls, ['client-1', 'client-2'], 'each client should be closed immediately after its call');
});

// Create in-memory MCP server with test tool
const createTestMcpClient = async () => {
  // eslint-disable-next-line @typescript-eslint/no-deprecated
  const server = new Server({ name: 'test-server', version: '1.0.0' }, { capabilities: { tools: {} } });

  // Register tools/list handler
  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [
      {
        name: 'testTool',
        description: 'A test tool',
        inputSchema: {
          type: 'object',
          properties: {
            testParam: { type: 'string', description: 'Test parameter' },
          },
          required: ['testParam'],
        },
      },
    ],
  }));

  // Register tools/call handler
  server.setRequestHandler(CallToolRequestSchema, async () => ({
    content: [{ type: 'text', text: 'ok' }],
  }));

  // Create linked in-memory transports
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await server.connect(serverTransport);

  // Return real AI SDK MCP client connected to in-memory server
  return createMCPClient({ transport: clientTransport });
};

test('configureWorker refuses to replace a different Activity with the same name', (t) => {
  const provider = new TestProvider(helloWorkflowGenerator());
  const files = async () => ({}) as any;
  const base = { taskQueue: 'tq' } as WorkerOptions;
  const duplicate = (name: string) => ({
    instanceOf: TypeError,
    message: new RegExp(`^Duplicate registration of Activity '${name}': AiSdkPlugin would replace`),
  });

  // A caller Activity, another instance's provider, or another instance's MCP
  // client factory under the same server name is refused.
  const callerActivities = { ...base, activities: { invokeModel: async () => 'caller' } };
  t.throws(() => new AiSdkPlugin({ modelProvider: provider }).configureWorker(callerActivities), duplicate('invokeModel'));
  const withMcp = new AiSdkPlugin({ modelProvider: provider, mcpClientFactories: { files } }).configureWorker(base);
  const otherProvider = new AiSdkPlugin({ modelProvider: new TestProvider(helloWorkflowGenerator()) });
  t.throws(() => otherProvider.configureWorker(withMcp), duplicate('invokeModel'));
  const otherFactory = new AiSdkPlugin({ modelProvider: provider, mcpClientFactories: { files: async () => ({}) as any } });
  t.throws(() => otherFactory.configureWorker(withMcp), duplicate('files-listTools'));
  const otherDownload = new AiSdkPlugin({ modelProvider: provider, download: localDownload });
  t.throws(() => otherDownload.configureWorker(withMcp), duplicate('invokeModel'));
  // Another operation built from the same provider, registered under this name.
  const aliased = { ...base, activities: { invokeModel: (createActivities(provider) as any).invokeEmbeddingModel } };
  t.throws(() => new AiSdkPlugin({ modelProvider: provider }).configureWorker(aliased), duplicate('invokeModel'));

  // configureWorker accepts the same instance twice (NativeConnection and
  // Worker), the same options twice, and createActivities with the same
  // provider. With workflowsPath, Worker.create still rejects a repeated
  // AiSdkPlugin in the bundler: each one prepends its polyfill entry.
  const plugin = new AiSdkPlugin({ modelProvider: provider, mcpClientFactories: { files } });
  t.notThrows(() => plugin.configureWorker(plugin.configureWorker(base)));
  t.notThrows(() => new AiSdkPlugin({ modelProvider: provider, mcpClientFactories: { files } }).configureWorker(withMcp));
  const callerSet = { ...base, activities: createActivities(provider) };
  t.notThrows(() => new AiSdkPlugin({ modelProvider: provider }).configureWorker(callerSet));
});

test('AiSdkPlugin reports the package version for worker heartbeats', (t) => {
  const plugin = new AiSdkPlugin({ modelProvider: new TestProvider(helloWorkflowGenerator()) });
  t.is(plugin.version, JSON.parse(readFileSync(path.join(__dirname, '../../package.json'), 'utf8')).version);
});

test('runWorker refuses an Activity that a later plugin replaced, before polling', async (t) => {
  const connection = await NativeConnection.connect({ address: t.context.env.address });
  // StrandsPlugin also registers invokeModel; this plugin stands in for it.
  const later = new SimplePlugin({ name: 'later', activities: { invokeModel: async () => 'other' } });
  const worker = await Worker.create({
    connection,
    namespace: t.context.env.namespace,
    taskQueue: 'test-ai-later-plugin',
    plugins: [new AiSdkPlugin({ modelProvider: new TestProvider(helloWorkflowGenerator()) }), later],
  });
  await t.throwsAsync(worker.run(), {
    instanceOf: TypeError,
    message: /^Duplicate registration of Activity 'invokeModel': a plugin applied after AiSdkPlugin replaced/,
  });
  t.is(worker.getState(), 'STOPPED');
  // Throws if the Worker still holds the connection.
  await connection.close();
});

test('runWorker starts the Worker at once, so runUntil stops it after its callback', async (t) => {
  const connection = await NativeConnection.connect({ address: t.context.env.address });
  // runUntil stops the Worker after its callback only if the Worker is already
  // running. A callback that throws synchronously finishes before run() yields.
  const alone = await Worker.create({
    connection,
    namespace: t.context.env.namespace,
    taskQueue: 'test-ai-run-until-throws',
    plugins: [new AiSdkPlugin({ modelProvider: new TestProvider(helloWorkflowGenerator()) })],
  });
  await t.throwsAsync(
    alone.runUntil(() => {
      throw new Error('boom');
    }),
    { message: 'boom' }
  );
  t.is(alone.getState(), 'STOPPED');
  // Each plugin's check runs before the Worker starts; none may yield.
  const options = { modelProvider: new TestProvider(helloWorkflowGenerator()) };
  const two = await Worker.create({
    connection,
    namespace: t.context.env.namespace,
    taskQueue: 'test-ai-run-until-two',
    plugins: [new AiSdkPlugin(options), new AiSdkPlugin(options)],
  });
  t.is(await two.runUntil(() => Promise.resolve('done')), 'done');
  t.is(two.getState(), 'STOPPED');
  await connection.close();
});

test('MCP tool schema survives activity serialization', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new AiSdkPlugin({
        modelProvider: new TestProvider(helloWorkflowGenerator()),
        mcpClientFactories: { testServer: createTestMcpClient },
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(mcpSchemaTestWorkflow);

    // These would FAIL with the bug (undefined), PASS with fix
    t.is(result.schemaType, 'object', 'Schema type should be "object", not undefined');
    t.true(result.hasProperties, 'Schema should have properties');
    t.is(result.propertyDescription, 'Test parameter', 'Property description should be preserved');
  });
});

// Currently fails in CI due to invalid server response but passes locally
test.skip('MCP Use', async (t) => {
  if (remoteTests) {
    t.timeout(120 * 1000);
  }
  const { createWorker, executeWorkflow } = helpers(t);

  const mcpClientFactories = {
    testServer: () =>
      createMCPClient({
        transport: new StdioClientTransport({
          command: 'npx',
          args: ['-y', '@modelcontextprotocol/server-filesystem@latest', __dirname],
        }),
      }),
  };
  const worker = await createWorker({
    plugins: [
      new AiSdkPlugin({
        modelProvider: remoteTests ? openai : new TestProvider(mcpGenerator()),
        mcpClientFactories,
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(mcpWorkflow, {
      args: ['Tell me what files you know about. Use your tools.'],
    });

    t.assert(result);
    if (!remoteTests) {
      t.is('Some files', result);
    }
  });
});

test('Captured multi-step tools history replays deterministically', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const plugin = () => new AiSdkPlugin({ modelProvider: new TestProvider(toolsWorkflowGenerator()) });

  const worker = await createWorker({ plugins: [plugin()], activities: { getWeather } });

  let history;
  await worker.runUntil(async () => {
    const handle = await startWorkflow(toolsWorkflow, {
      args: ['What is the weather in Tokyo?'],
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(await handle.result(), 'Test weather result');
    history = await handle.fetchHistory();
  });

  // Replay the captured history with a fresh plugin instance. Any non-determinism
  // in the ai@7 agent loop or the sandbox polyfills would surface as
  // DeterminismViolationError.
  await CoreWorker.runReplayHistory({ workflowBundle: t.context.workflowBundle, plugins: [plugin()] }, history!);
  t.pass();
});

function* mcpReplayGenerator(): Generator<ModelResponse> {
  yield toolCallResponse('testTool', '{"testParam":"hello"}');
  yield textResponse('Done');
}

test('Captured multi-step MCP history replays deterministically', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const plugin = () =>
    new AiSdkPlugin({
      modelProvider: new TestProvider(mcpReplayGenerator()),
      mcpClientFactories: { testServer: createTestMcpClient },
    });

  const worker = await createWorker({ plugins: [plugin()] });

  let history;
  await worker.runUntil(async () => {
    const handle = await startWorkflow(mcpWorkflow, {
      args: ['What files do you have? Use your tools.'],
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(await handle.result(), 'Done');
    history = await handle.fetchHistory();
  });

  await CoreWorker.runReplayHistory({ workflowBundle: t.context.workflowBundle, plugins: [plugin()] }, history!);
  t.pass();
});
