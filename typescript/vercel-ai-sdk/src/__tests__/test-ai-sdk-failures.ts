import type {
  EmbeddingModelV4,
  EmbeddingModelV4Result,
  ImageModelV4,
  LanguageModelV4,
  LanguageModelV4GenerateResult,
  LanguageModelV4StreamResult,
  ProviderV4,
} from '@ai-sdk/provider';
import { APICallError } from '@ai-sdk/provider';
import { WorkflowFailedError } from '@temporalio/client';
import { ActivityFailure, ApplicationFailure, TemporalFailure } from '@temporalio/common';
import { temporal } from '@temporalio/proto';
import { helpers } from '@temporalio/test-helpers';
import { MockActivityEnvironment } from '@temporalio/testing';
import { AiSdkPlugin, createActivities } from '..';
import { makeAiSdkTestFunction } from './helpers/ai-sdk';
import type { ModelResponse } from './stubs/ai-sdk';
import { embeddingResponse, textResponse } from './stubs/ai-sdk';
import { helloWorldAgent } from './workflows/ai-sdk';
import EventType = temporal.api.enums.v1.EventType;

const test = makeAiSdkTestFunction();

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
