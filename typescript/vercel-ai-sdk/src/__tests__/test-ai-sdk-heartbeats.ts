import type {
  EmbeddingModelV4,
  EmbeddingModelV4Result,
  ImageModelV4,
  LanguageModelV4,
  LanguageModelV4GenerateResult,
  LanguageModelV4StreamResult,
  ProviderV4,
} from '@ai-sdk/provider';
import { optionalTsToMs } from '@temporalio/common/lib/time';
import { temporal } from '@temporalio/proto';
import { helpers } from '@temporalio/test-helpers';
import { AiSdkPlugin } from '..';
import { makeAiSdkTestFunction } from './helpers/ai-sdk';
import { createFinishReason, createUsage, embeddingResponse, textResponse } from './stubs/ai-sdk';
import { slowModelWorkflow } from './workflows/ai-sdk';
import EventType = temporal.api.enums.v1.EventType;

const test = makeAiSdkTestFunction();

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
