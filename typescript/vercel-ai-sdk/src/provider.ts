import type { ReadableStreamDefaultController } from 'node:stream/web';
import type {
  EmbeddingModelV4,
  EmbeddingModelV4CallOptions,
  EmbeddingModelV4Result,
  ImageModelV4,
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4GenerateResult,
  LanguageModelV4StreamResult,
  ProviderV4,
} from '@ai-sdk/provider';
import * as workflow from '@temporalio/workflow';
import type { ActivityOptions } from '@temporalio/workflow';
import { ApplicationFailure } from '@temporalio/common';
import type { Duration } from '@temporalio/common/lib/time';
import { encodeCallOptionsFileData } from './file-data';

// An older Event History can hold an Activity that `experimental_download` scheduled for a
// URL. Such a history replays only if the model still reports that it supports no URL.
const DOWNLOAD_IN_ACTIVITY_PATCH = 'vercel-ai-sdk-download-urls-in-model-activity';

// `ReadableStream` is a sandbox global; type-only import keeps `node:stream/web`
// out of the workflow bundle (es2023 lib has no DOM types).
declare const ReadableStream: typeof import('node:stream/web').ReadableStream;

function abortReason(signal: AbortSignal): unknown {
  const reason: unknown = signal.reason;
  if (reason instanceof Error || typeof reason !== 'object' || reason === null) return reason;
  const { name, message } = reason as { name?: unknown; message?: unknown };
  if (name !== 'AbortError' && name !== 'TimeoutError') return reason;
  // The sandbox `AbortController` comes from the host realm, so its reason fails the
  // `instanceof Error` check in the AI SDK's `isAbortError`. Rebuild it in this realm.
  return Object.assign(new Error(String(message)), { name });
}

// An older Event History can hold a model call that ran to its end after its signal
// aborted. Such a history replays only if the abort still sends no command.
const ABORT_CANCELS_ACTIVITY_PATCH = 'vercel-ai-sdk-abort-signal-cancels-model-activity';

// An `AbortSignal` can't cross the payload converter, so the caller's signal cancels the
// Activity from the Workflow instead.
//
// Not an `async` function. A call that cancels nothing must resume the Workflow at the
// same point as a direct Activity call, or its commands can change order on replay.
function runAbortable<T>(signal: AbortSignal | undefined, fn: () => Promise<T>): Promise<T> {
  if (signal === undefined || !workflow.patched(ABORT_CANCELS_ACTIVITY_PATCH)) return fn();
  return cancelOnAbort(signal, fn);
}

async function cancelOnAbort<T>(signal: AbortSignal, fn: () => Promise<T>): Promise<T> {
  if (signal.aborted) throw abortReason(signal);
  const scope = new workflow.CancellationScope();
  const onAbort = (): void => scope.cancel();
  signal.addEventListener('abort', onAbort, { once: true });
  try {
    return await scope.run(fn);
  } catch (err) {
    if (signal.aborted && workflow.isCancellation(err)) throw abortReason(signal);
    throw err;
  } finally {
    signal.removeEventListener('abort', onAbort);
  }
}

/**
 * Options for configuring the TemporalProvider with per-model activity settings.
 */
export interface TemporalProviderOptions {
  /**
   * Default activity options applied to all model types.
   * These can be overridden by model-specific options.
   */
  default?: ActivityOptions;

  /**
   * Activity options specific to language model calls.
   * Merged with default options, with these taking precedence.
   */
  languageModel?: ActivityOptions & {
    /**
     * Topic name on the workflow's stream that streaming model calls publish
     * raw stream parts to. When set, `doStream` is enabled and routes through
     * the streaming activity; when unset, `doStream` throws. Pick a unique
     * name per concurrent streaming call to keep event streams separable.
     */
    streamingTopic?: string;

    /**
     * Batch interval for the per-activity `WorkflowStreamClient` that
     * publishes stream parts back to the workflow. Lower values reduce
     * latency at the cost of more signal traffic. Defaults to 100ms.
     */
    streamingBatchInterval?: Duration;
  };

  /**
   * Activity options specific to embedding model calls.
   * Merged with default options, with these taking precedence.
   */
  embeddingModel?: ActivityOptions & {
    /**
     * Largest number of values that `embedMany` sends to one `invokeEmbeddingModel` Activity.
     * When set, `embedMany` splits the values into chunks of this size and runs one Activity per
     * chunk, in parallel up to its `maxParallelCalls`. Unset sends all values to one Activity.
     * Changing it in a deployed Workflow needs Workflow versioning.
     *
     * Each chunk gets the full `providerOptions`. The model that can cut per-value options
     * to a chunk is on the Worker, and it needs all the values to do so. To send them to
     * each Activity would put the full input in Event History one time for each chunk.
     */
    maxEmbeddingsPerCall?: number;
  };
}

/**
 * A language model implementation that delegates AI model calls to Temporal activities.
 * This allows workflows to invoke AI models through the Temporal execution model.
 */
export class TemporalLanguageModel implements LanguageModelV4 {
  readonly specificationVersion = 'v4';
  readonly provider = 'temporal';
  private readonly streamingTopic: string | undefined;
  private readonly streamingBatchInterval: Duration | undefined;

  constructor(
    readonly modelId: string,
    readonly options?: ActivityOptions & { streamingTopic?: string; streamingBatchInterval?: Duration }
  ) {
    this.streamingTopic = options?.streamingTopic;
    this.streamingBatchInterval = options?.streamingBatchInterval;
  }

  // The Workflow sandbox has no network. URL parts go to the model Activity, which downloads
  // the ones the real model does not support.
  get supportedUrls(): Record<string, RegExp[]> {
    return workflow.patched(DOWNLOAD_IN_ACTIVITY_PATCH) ? { '*': [/.*/] } : {};
  }

  async doGenerate(options: LanguageModelV4CallOptions): Promise<LanguageModelV4GenerateResult> {
    const activities = workflow.proxyActivities({
      startToCloseTimeout: '10 minutes',
      ...this.options,
      heartbeatTimeout: this.options?.heartbeatTimeout ?? '10 seconds',
    });
    const { abortSignal, ...callOptions } = options;
    const result = await runAbortable(abortSignal, () =>
      activities.invokeModel!({
        modelId: this.modelId,
        options: encodeCallOptionsFileData(callOptions),
      })
    );
    if (result === undefined) {
      throw ApplicationFailure.nonRetryable('Received undefined response from model activity.');
    }
    if (result.response !== undefined && result.response.timestamp) {
      const timestamp = new Date(result.response.timestamp);
      // Only set if it's a valid date
      if (!isNaN(timestamp.getTime())) {
        result.response.timestamp = timestamp;
      }
    }
    return result;
  }

  async doStream(options: LanguageModelV4CallOptions): Promise<LanguageModelV4StreamResult> {
    if (this.streamingTopic === undefined) {
      throw ApplicationFailure.nonRetryable(
        'Streaming not enabled. Set streamingTopic in languageModel provider options.'
      );
    }

    // Call the streaming activity, which publishes tokens via stream
    // and returns the accumulated result.
    const activities = workflow.proxyActivities({
      startToCloseTimeout: '10 minutes',
      ...this.options,
      heartbeatTimeout: this.options?.heartbeatTimeout ?? '10 seconds',
    });
    const { abortSignal, ...callOptions } = options;
    const result = await runAbortable(abortSignal, () =>
      activities.invokeModelStreaming!({
        modelId: this.modelId,
        options: encodeCallOptionsFileData(callOptions),
        streamingTopic: this.streamingTopic,
        streamingBatchInterval: this.streamingBatchInterval,
      })
    );
    if (result === undefined) {
      throw ApplicationFailure.nonRetryable('Received undefined response from streaming model activity.');
    }

    // Wrap the accumulated result as a ReadableStream that replays the content.
    // Real-time token streaming already happened via stream in the activity.
    const stream = new ReadableStream({
      start(controller: ReadableStreamDefaultController) {
        controller.enqueue({ type: 'stream-start', warnings: result.warnings ?? [] });
        if (result.response !== undefined) {
          controller.enqueue({
            type: 'response-metadata',
            id: result.response.id,
            modelId: result.response.modelId,
            timestamp: result.response.timestamp ? new Date(result.response.timestamp) : undefined,
          });
        }
        let partIndex = 0;
        for (const item of result.content ?? []) {
          const id = `part-${partIndex++}`;
          if (item.type === 'text') {
            controller.enqueue({ type: 'text-start', id, providerMetadata: item.providerMetadata });
            controller.enqueue({ type: 'text-delta', id, delta: item.text });
            controller.enqueue({ type: 'text-end', id });
          } else if (item.type === 'reasoning') {
            controller.enqueue({ type: 'reasoning-start', id });
            controller.enqueue({ type: 'reasoning-delta', id, delta: item.text });
            controller.enqueue({ type: 'reasoning-end', id });
          } else {
            controller.enqueue(item);
          }
        }
        controller.enqueue({
          type: 'finish',
          finishReason: result.finishReason,
          usage: result.usage,
          providerMetadata: result.providerMetadata,
        });
        controller.close();
      },
    });

    return { stream, request: result.request, response: result.response };
  }
}

/**
 * An embedding model implementation that delegates embedding generation to Temporal activities.
 * This allows workflows to generate embeddings through the Temporal execution model.
 */
export class TemporalEmbeddingModel implements EmbeddingModelV4 {
  readonly specificationVersion = 'v4';
  readonly provider = 'temporal';
  /**
   * The `maxEmbeddingsPerCall` provider option. Undefined unless set, so `embedMany` sends all
   * values to one Activity.
   */
  readonly maxEmbeddingsPerCall: number | undefined;
  /**
   * Indicates the underlying embedding model API can handle concurrent requests.
   * Set to true since we delegate to the actual provider which manages its own concurrency.
   */
  readonly supportsParallelCalls = true;

  constructor(
    readonly modelId: string,
    readonly options?: ActivityOptions & { maxEmbeddingsPerCall?: number }
  ) {
    this.maxEmbeddingsPerCall = options?.maxEmbeddingsPerCall;
  }

  async doEmbed(options: EmbeddingModelV4CallOptions): Promise<EmbeddingModelV4Result> {
    const { maxEmbeddingsPerCall: _maxEmbeddingsPerCall, ...activityOptions } = this.options ?? {};
    const activities = workflow.proxyActivities({
      startToCloseTimeout: '10 minutes',
      ...activityOptions,
      heartbeatTimeout: activityOptions.heartbeatTimeout ?? '10 seconds',
    });
    const result = await runAbortable(options.abortSignal, () =>
      activities.invokeEmbeddingModel!({
        modelId: this.modelId,
        values: options.values,
        providerOptions: options.providerOptions,
        headers: options.headers,
      })
    );
    if (result === undefined) {
      throw ApplicationFailure.nonRetryable('Received undefined response from embedding model activity.');
    }
    return result;
  }
}

/**
 * A Temporal-specific provider implementation that creates AI models which execute
 * through Temporal activities. This provider integrates AI SDK models with Temporal's
 * execution model to ensure reliable, durable AI model invocations.
 */
export class TemporalProvider implements ProviderV4 {
  readonly specificationVersion = 'v4';

  constructor(readonly options?: TemporalProviderOptions) {}

  imageModel(_modelId: string): ImageModelV4 {
    throw new Error('Not implemented');
  }

  languageModel(modelId: string): LanguageModelV4 {
    return new TemporalLanguageModel(modelId, {
      ...this.options?.default,
      ...this.options?.languageModel,
    });
  }

  embeddingModel(modelId: string): EmbeddingModelV4 {
    return new TemporalEmbeddingModel(modelId, {
      ...this.options?.default,
      ...this.options?.embeddingModel,
    });
  }
}

/**
 * A singleton instance of TemporalProvider for convenient use in applications.
 */
export const temporalProvider: TemporalProvider = new TemporalProvider();
