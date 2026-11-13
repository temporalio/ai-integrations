import { openai } from '@ai-sdk/openai';
import type {
  EmbeddingModelV4,
  ImageModelV4,
  LanguageModelV4,
  LanguageModelV4GenerateResult,
  LanguageModelV4StreamPart,
  LanguageModelV4StreamResult,
  ProviderV4,
} from '@ai-sdk/provider';
import { temporal } from '@temporalio/proto';
import { helpers } from '@temporalio/test-helpers';
import { AiSdkPlugin } from '..';
import { getWeather } from './activities/ai-sdk';
import { makeAiSdkTestFunction, remoteTests } from './helpers/ai-sdk';
import type { ModelResponse } from './stubs/ai-sdk';
import {
  createFinishReason,
  createUsage,
  embeddingGenerator,
  helloWorkflowGenerator,
  TestProvider,
  textResponse,
  toolsWorkflowGenerator,
} from './stubs/ai-sdk';
import {
  chunkedEmbeddingWorkflow,
  embeddingWorkflow,
  generateObjectWorkflow,
  helloWorldAgent,
  middlewareWorkflow,
  streamingMetadataWorkflow,
  toolsWorkflow,
} from './workflows/ai-sdk';
import EventType = temporal.api.enums.v1.EventType;

const test = makeAiSdkTestFunction();

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

class EchoEmbeddingProvider implements ProviderV4 {
  readonly specificationVersion = 'v4';

  languageModel(_modelId: string): LanguageModelV4 {
    throw new Error('Not implemented.');
  }

  embeddingModel(_modelId: string): EmbeddingModelV4 {
    return {
      specificationVersion: 'v4',
      provider: 'temporal-test',
      modelId: 'echo',
      maxEmbeddingsPerCall: undefined,
      supportsParallelCalls: true,
      async doEmbed({ values }) {
        return { embeddings: values.map((value) => [Number(value)]), warnings: [] };
      },
    };
  }

  imageModel(_modelId: string): ImageModelV4 {
    throw new Error('Not implemented.');
  }
}

function embeddingActivityInputs(events: temporal.api.history.v1.IHistoryEvent[]): string[][] {
  return events
    .filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED)
    .map((e) => {
      const data = e.activityTaskScheduledEventAttributes!.input!.payloads![0]!.data!;
      return JSON.parse(Buffer.from(data).toString()).values;
    });
}

test('maxEmbeddingsPerCall splits embedMany into one Activity per chunk', async (t) => {
  const values = ['0', '1', '2', '3', '4'];
  const { createWorker, startWorkflow } = helpers(t);
  const worker = await createWorker({
    plugins: [new AiSdkPlugin({ modelProvider: new EchoEmbeddingProvider() })],
  });

  await worker.runUntil(async () => {
    const chunked = await startWorkflow(chunkedEmbeddingWorkflow, { args: [values, 2] });
    t.deepEqual(await chunked.result(), [[0], [1], [2], [3], [4]]);
    const chunkedInputs = embeddingActivityInputs((await chunked.fetchHistory()).events!);
    t.deepEqual(chunkedInputs, [['0', '1'], ['2', '3'], ['4']]);

    const unset = await startWorkflow(chunkedEmbeddingWorkflow, { args: [values] });
    t.deepEqual(await unset.result(), [[0], [1], [2], [3], [4]]);
    t.deepEqual(embeddingActivityInputs((await unset.fetchHistory()).events!), [values]);
  });
});
