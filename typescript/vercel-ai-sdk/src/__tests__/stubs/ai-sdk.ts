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
  LanguageModelV4Usage,
  ProviderV4,
} from '@ai-sdk/provider';
import { Context } from '@temporalio/activity';

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

export function createFinishReason(
  unified: 'stop' | 'length' | 'content-filter' | 'tool-calls' | 'error' | 'other'
): LanguageModelV4FinishReason {
  return { unified, raw: undefined };
}

export function createUsage(inputTokens: number = 10, outputTokens: number = 20): LanguageModelV4Usage {
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

export function contentResponse(
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

export function textResponse(content: string): ModelResponse {
  return contentResponse([
    {
      type: 'text',
      text: content,
    },
  ]);
}

export function toolCallResponse(toolName: string, input: string): ModelResponse {
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

export function embeddingResponse(values: string[]): EmbeddingResponse {
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

export function* helloWorkflowGenerator(): Generator<ModelResponse> {
  yield textResponse('Test Haiku');
}

export function* embeddingGenerator(): Generator<EmbeddingResponse> {
  yield embeddingResponse(['Hello, world!', 'How are you?']);
}

export function* toolsWorkflowGenerator(): Generator<ModelResponse> {
  yield toolCallResponse('getWeather', '{"location":"Tokyo"}');
  yield textResponse('Test weather result');
}

export class RecordingProvider implements ProviderV4 {
  readonly specificationVersion = 'v4';
  readonly calls: LanguageModelV4CallOptions[] = [];

  constructor(
    private readonly supportedUrls: Record<string, RegExp[]>,
    private readonly request?: { body: unknown }
  ) {}

  languageModel(_modelId: string): LanguageModelV4 {
    const { calls, request, supportedUrls } = this;
    return {
      specificationVersion: 'v4',
      provider: 'temporal-test',
      modelId: 'recording-test-model',
      supportedUrls,
      async doGenerate(options): Promise<LanguageModelV4GenerateResult> {
        calls.push(options);
        return { ...textResponse('seen'), request };
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

// Records the `abortSignal` each model call receives. When `block` is set, a call waits until
// that signal aborts, heartbeating so the Activity can receive cancellation.
export class AbortRecordingProvider implements ProviderV4 {
  readonly specificationVersion = 'v4';
  readonly signals: unknown[] = [];
  started = false;
  providerAborted = false;

  constructor(private readonly block: boolean) {}

  private async callModel(signal: AbortSignal | undefined): Promise<void> {
    this.signals.push(signal);
    this.started = true;
    if (!this.block) return;
    while (!signal!.aborted) {
      Context.current().heartbeat();
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    this.providerAborted = true;
    throw signal!.reason;
  }

  languageModel(_modelId: string): LanguageModelV4 {
    const callModel = this.callModel.bind(this);
    return {
      specificationVersion: 'v4',
      provider: 'temporal-test',
      modelId: 'abort-test-model',
      supportedUrls: {},
      async doGenerate(options): Promise<LanguageModelV4GenerateResult> {
        await callModel(options.abortSignal);
        return textResponse('not aborted');
      },
      async doStream(options): Promise<LanguageModelV4StreamResult> {
        await callModel(options.abortSignal);
        return { stream: new ReadableStream({ start: (controller) => controller.close() }) };
      },
    };
  }

  embeddingModel(_modelId: string): EmbeddingModelV4 {
    const callModel = this.callModel.bind(this);
    return {
      specificationVersion: 'v4',
      provider: 'temporal-test',
      modelId: 'abort-test-embedding-model',
      maxEmbeddingsPerCall: undefined,
      supportsParallelCalls: true,
      async doEmbed(options): Promise<EmbeddingModelV4Result> {
        await callModel(options.abortSignal);
        return embeddingResponse(options.values);
      },
    };
  }

  imageModel(_modelId: string): ImageModelV4 {
    throw new Error('Not implemented.');
  }
}
