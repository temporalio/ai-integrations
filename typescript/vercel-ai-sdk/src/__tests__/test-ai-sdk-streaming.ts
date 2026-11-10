/**
 * Unit tests for the streaming model activity (`invokeModelStreaming`).
 *
 * Runs the activity in a MockActivityEnvironment against a model that yields a
 * scripted V4 stream-part sequence, with a fake Temporal client that records the
 * stream parts published to the workflow-stream topic.
 */
import test from 'ava';
import type {
  EmbeddingModelV4,
  ImageModelV4,
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4GenerateResult,
  LanguageModelV4StreamPart,
  LanguageModelV4StreamResult,
  LanguageModelV4Usage,
  ProviderV4,
} from '@ai-sdk/provider';
import { APICallError } from '@ai-sdk/provider';
import type { Client } from '@temporalio/client';
import { ApplicationFailure, defaultPayloadConverter } from '@temporalio/common';
import { decodePayloadWire } from '@temporalio/workflow-streams/workflow';
import { MockActivityEnvironment } from '@temporalio/testing';
import { createActivities } from '../activities';
import type { InvokeModelResult, InvokeModelStreamingArgs } from '../activities';

class StreamingTestModel implements LanguageModelV4 {
  readonly specificationVersion = 'v4';
  readonly provider = 'temporal';
  readonly modelId = 'StreamingTestModel';

  constructor(private readonly parts: LanguageModelV4StreamPart[]) {}

  get supportedUrls(): Record<string, RegExp[]> {
    return {};
  }

  async doGenerate(_options: LanguageModelV4CallOptions): Promise<LanguageModelV4GenerateResult> {
    throw new Error('Generate not supported.');
  }

  async doStream(_options: LanguageModelV4CallOptions): Promise<LanguageModelV4StreamResult> {
    const parts = this.parts;
    return {
      stream: new ReadableStream<LanguageModelV4StreamPart>({
        start(controller) {
          for (const part of parts) controller.enqueue(part);
          controller.close();
        },
      }),
      request: { body: 'test-request-body' },
      response: { headers: { 'x-test-header': 'yes' } },
    };
  }
}

function streamingProvider(parts: LanguageModelV4StreamPart[]): ProviderV4 {
  return {
    specificationVersion: 'v4',
    languageModel: (_modelId: string): LanguageModelV4 => new StreamingTestModel(parts),
    embeddingModel: (_modelId: string): EmbeddingModelV4 => {
      throw new Error('Not implemented');
    },
    imageModel: (_modelId: string): ImageModelV4 => {
      throw new Error('Not implemented');
    },
  };
}

/**
 * Minimal fake of the Temporal client surface that WorkflowStreamClient uses:
 * records every published batch instead of signaling a workflow.
 */
function recordingClient(published: unknown[][]): Client {
  return {
    withAbortSignal: <R>(_signal: AbortSignal, fn: () => Promise<R>) => fn(),
    workflow: {
      getHandle: (workflowId: string) => ({
        workflowId,
        async signal(_name: string, input: { items: unknown[] }) {
          published.push(input.items);
        },
      }),
    },
  } as unknown as Client;
}

async function runStreamingActivity(
  parts: LanguageModelV4StreamPart[]
): Promise<{ result: InvokeModelResult; published: unknown[][] }> {
  const activities = createActivities(streamingProvider(parts)) as {
    invokeModelStreaming: (args: InvokeModelStreamingArgs) => Promise<InvokeModelResult>;
  };
  const published: unknown[][] = [];
  const env = new MockActivityEnvironment(undefined, { client: recordingClient(published) });
  const result = (await env.run(activities.invokeModelStreaming, {
    modelId: 'StreamingTestModel',
    options: { prompt: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }] },
    streamingTopic: 'test-topic',
    streamingBatchInterval: '10ms',
  } satisfies InvokeModelStreamingArgs)) as InvokeModelResult;
  return { result, published };
}

const usage: LanguageModelV4Usage = {
  inputTokens: { total: 10, noCache: undefined, cacheRead: undefined, cacheWrite: undefined },
  outputTokens: { total: 20, text: undefined, reasoning: undefined },
};

test('invokeModelStreaming publishes every part and assembles the final result', async (t) => {
  const timestamp = new Date('2026-01-01T00:00:00Z');
  const parts: LanguageModelV4StreamPart[] = [
    { type: 'stream-start', warnings: [{ type: 'unsupported', feature: 'test-feature' }] },
    { type: 'response-metadata', id: 'resp-1', modelId: 'test-model-id', timestamp },
    { type: 'text-start', id: 'txt-1' },
    { type: 'text-delta', id: 'txt-1', delta: 'Hello, ' },
    { type: 'text-delta', id: 'txt-1', delta: 'world!' },
    { type: 'text-end', id: 'txt-1' },
    { type: 'reasoning-start', id: 'rsn-1' },
    { type: 'reasoning-delta', id: 'rsn-1', delta: 'pondering' },
    { type: 'reasoning-end', id: 'rsn-1' },
    { type: 'tool-call', toolCallId: 'call-1', toolName: 'getWeather', input: '{"location":"Tokyo"}' },
    { type: 'raw', rawValue: { provider: 'specific' } },
    { type: 'finish', finishReason: { unified: 'stop', raw: undefined }, usage },
  ];

  const { result, published } = await runStreamingActivity(parts);

  t.deepEqual(result.content, [
    { type: 'text', text: 'Hello, world!', providerMetadata: undefined },
    { type: 'reasoning', text: 'pondering', providerMetadata: undefined },
    { type: 'tool-call', toolCallId: 'call-1', toolName: 'getWeather', input: '{"location":"Tokyo"}' },
  ]);
  t.deepEqual(result.finishReason, { unified: 'stop', raw: undefined });
  t.deepEqual(result.usage, usage);
  t.deepEqual(result.warnings, [{ type: 'unsupported', feature: 'test-feature' }]);
  t.deepEqual(result.request, { body: 'test-request-body' });
  // Response metadata from the stream is merged with the model's response info.
  t.deepEqual(result.response, {
    id: 'resp-1',
    modelId: 'test-model-id',
    timestamp,
    headers: { 'x-test-header': 'yes' },
  });

  // Every stream part (including `raw`) is published to the topic exactly once.
  const totalPublished = published.reduce((sum, batch) => sum + batch.length, 0);
  t.is(totalPublished, parts.length);
});

test('invokeModelStreaming retains the latest text provider metadata', async (t) => {
  const startMetadata = { test: { source: 'start' } };
  const deltaMetadata = { test: { source: 'delta' } };
  const endMetadata = { test: { source: 'end' } };
  const parts: LanguageModelV4StreamPart[] = [
    { type: 'text-start', id: 'start-only', providerMetadata: startMetadata },
    { type: 'text-delta', id: 'start-only', delta: 'first' },
    { type: 'text-end', id: 'start-only' },
    { type: 'text-start', id: 'updated-by-delta', providerMetadata: startMetadata },
    { type: 'text-delta', id: 'updated-by-delta', delta: 'second', providerMetadata: deltaMetadata },
    { type: 'text-end', id: 'updated-by-delta' },
    { type: 'text-start', id: 'updated-by-end', providerMetadata: startMetadata },
    { type: 'text-delta', id: 'updated-by-end', delta: 'third', providerMetadata: deltaMetadata },
    { type: 'text-end', id: 'updated-by-end', providerMetadata: endMetadata },
    { type: 'finish', finishReason: { unified: 'stop', raw: undefined }, usage },
  ];

  const { result } = await runStreamingActivity(parts);

  t.deepEqual(result.content, [
    { type: 'text', text: 'first', providerMetadata: startMetadata },
    { type: 'text', text: 'second', providerMetadata: deltaMetadata },
    { type: 'text', text: 'third', providerMetadata: endMetadata },
  ]);
});

test('invokeModelStreaming throws when the stream emits an error and no finish part', async (t) => {
  const parts: LanguageModelV4StreamPart[] = [
    { type: 'stream-start', warnings: [] },
    { type: 'text-start', id: 'txt-1' },
    { type: 'text-delta', id: 'txt-1', delta: 'partial' },
    { type: 'error', error: 'connection reset' },
  ];

  const err = await t.throwsAsync(runStreamingActivity(parts), { instanceOf: ApplicationFailure });
  t.false(err!.nonRetryable);
  t.regex(err!.message, /error part and ended without a finish part/);
  t.regex(err!.message, /connection reset/);
});

test('invokeModelStreaming throws when the stream emits an error even if a finish part follows', async (t) => {
  const parts: LanguageModelV4StreamPart[] = [
    { type: 'stream-start', warnings: [] },
    { type: 'text-start', id: 'txt-1' },
    { type: 'text-delta', id: 'txt-1', delta: 'partial' },
    { type: 'text-end', id: 'txt-1' },
    { type: 'error', error: 'mid-stream provider failure' },
    { type: 'finish', finishReason: { unified: 'error', raw: undefined }, usage },
  ];

  const err = await t.throwsAsync(runStreamingActivity(parts), { instanceOf: ApplicationFailure });
  t.false(err!.nonRetryable);
  t.regex(err!.message, /mid-stream provider failure/);
  t.notRegex(err!.message, /without a finish part/);
});

function errorPartStream(error: unknown): LanguageModelV4StreamPart[] {
  return [
    { type: 'stream-start', warnings: [] },
    { type: 'error', error },
    { type: 'finish', finishReason: { unified: 'error', raw: undefined }, usage },
  ];
}

test('invokeModelStreaming fails without retry on a non-retryable APICallError part', async (t) => {
  const error = new APICallError({
    message: 'Invalid API key',
    url: 'https://provider.invalid/v1/chat',
    requestBodyValues: {},
    statusCode: 401,
    isRetryable: false,
  });

  const err = await t.throwsAsync(runStreamingActivity(errorPartStream(error)), {
    instanceOf: ApplicationFailure,
  });
  t.true(err!.nonRetryable);
  t.is(err!.type, 'Error');
  t.is(err!.cause, error);
  t.regex(err!.message, /Invalid API key/);
});

test('A retryable APICallError stream part keeps the failure type Error', async (t) => {
  const error = new APICallError({
    message: 'Overloaded',
    url: 'https://provider.invalid/v1/chat',
    requestBodyValues: {},
    statusCode: 529,
  });

  const err = await t.throwsAsync(runStreamingActivity(errorPartStream(error)), {
    instanceOf: ApplicationFailure,
  });
  t.false(err!.nonRetryable);
  t.is(err!.type, 'Error');
});

test('invokeModelStreaming reports a plain object error part as JSON', async (t) => {
  const error = { type: 'overloaded_error', message: 'Overloaded' };
  const err = await t.throwsAsync(runStreamingActivity(errorPartStream(error)), {
    instanceOf: ApplicationFailure,
  });
  t.false(err!.nonRetryable);
  t.notRegex(err!.message, /\[object Object\]/);
  t.regex(err!.message, /: \{"type":"overloaded_error","message":"Overloaded"\}$/);
  t.is(err!.cause, undefined);
});

test('invokeModelStreaming retains finish provider metadata', async (t) => {
  const providerMetadata = { test: { cachedTokens: 7 } };
  const { result } = await runStreamingActivity([
    { type: 'finish', finishReason: { unified: 'stop', raw: undefined }, usage, providerMetadata },
  ]);

  t.deepEqual(result.providerMetadata, providerMetadata);
});

test('invokeModelStreaming publishes and returns binary file data as base64', async (t) => {
  const bytes = new Uint8Array([137, 80, 78, 71, 0, 255]);
  const encoded = { type: 'data', data: Buffer.from(bytes).toString('base64') } as const;
  const { result, published } = await runStreamingActivity([
    { type: 'file', mediaType: 'image/png', data: { type: 'data', data: bytes } },
    { type: 'reasoning-file', mediaType: 'image/png', data: { type: 'data', data: bytes } },
    { type: 'finish', finishReason: { unified: 'stop', raw: undefined }, usage },
  ]);

  t.deepEqual(result.content, [
    { type: 'file', mediaType: 'image/png', data: encoded },
    { type: 'reasoning-file', mediaType: 'image/png', data: encoded },
  ]);
  const decoder = new TextDecoder();
  const parts = published.flat().map((item) => {
    const payload = decodePayloadWire((item as { data: string }).data);
    return JSON.parse(decoder.decode(defaultPayloadConverter.fromPayload<Uint8Array>(payload)));
  });
  t.deepEqual(parts.slice(0, 2), result.content);
test('invokeModelStreaming downloads tool-result URLs and restores assistant URLs', async (t) => {
  const seen: LanguageModelV4CallOptions[] = [];
  const model = new StreamingTestModel([
    { type: 'finish', finishReason: { unified: 'stop', raw: undefined }, usage },
  ]);
  const provider: ProviderV4 = {
    ...streamingProvider([]),
    languageModel: (): LanguageModelV4 => ({
      specificationVersion: 'v4',
      provider: 'temporal',
      modelId: 'StreamingTestModel',
      // A `PromiseLike`, which the spec allows.
      supportedUrls: Promise.resolve({ 'image/*': [/^https:\/\/supported\.test\//] }),
      doGenerate: (options) => model.doGenerate(options),
      doStream: async (options) => {
        seen.push(options);
        return model.doStream(options);
      },
    }),
  };
  const downloaded: string[] = [];
  const activities = createActivities(provider, undefined, {
    download: async ({ url }) => {
      downloaded.push(url.href);
      return { data: new Uint8Array([1, 2, 3]), mediaType: 'image/gif' };
    },
  }) as { invokeModelStreaming: (args: InvokeModelStreamingArgs) => Promise<InvokeModelResult> };
  // URL fields arrive as strings, as they do after the JSON payload converter.
  const fileUrl = (url: string, mediaType: string) => ({
    type: 'file' as const,
    mediaType,
    data: { type: 'url' as const, url: url as unknown as URL },
  });
  const prompt: LanguageModelV4CallOptions['prompt'] = [
    { role: 'assistant', content: [fileUrl('https://unsupported.test/a.png', 'image/png')] },
    {
      role: 'tool',
      content: [
        {
          type: 'tool-result',
          toolCallId: 'call-1',
          toolName: 'screenshot',
          output: {
            type: 'content',
            value: [
              fileUrl('https://unsupported.test/b', 'image'),
              fileUrl('https://supported.test/c.png', 'image/png'),
            ],
          },
        },
      ],
    },
  ];
  const env = new MockActivityEnvironment(undefined, { client: recordingClient([]) });
  await env.run(activities.invokeModelStreaming, {
    modelId: 'StreamingTestModel',
    options: { prompt },
    streamingTopic: 'test-topic',
    streamingBatchInterval: '10ms',
  } satisfies InvokeModelStreamingArgs);

  t.deepEqual(downloaded, ['https://unsupported.test/b']);
  const [assistant, tool] = seen[0]!.prompt;
  if (assistant?.role !== 'assistant' || tool?.role !== 'tool') {
    throw new Error('Unexpected prompt');
  }
  const assistantFile = assistant.content[0];
  if (assistantFile?.type !== 'file' || assistantFile.data.type !== 'url') {
    throw new Error('Expected an assistant URL file part');
  }
  t.true(assistantFile.data.url instanceof URL);
  const toolResult = tool.content[0];
  if (toolResult?.type !== 'tool-result' || toolResult.output.type !== 'content') {
    throw new Error('Expected tool-result content');
  }
  const [downloadedFile, supportedFile] = toolResult.output.value;
  t.deepEqual(downloadedFile, {
    type: 'file',
    mediaType: 'image/gif',
    data: { type: 'data', data: new Uint8Array([1, 2, 3]) },
  });
  if (supportedFile?.type !== 'file' || supportedFile.data.type !== 'url') {
    throw new Error('Expected a supported URL file part');
  }
  t.true(supportedFile.data.url instanceof URL);
  t.is(supportedFile.data.url.href, 'https://supported.test/c.png');
});
