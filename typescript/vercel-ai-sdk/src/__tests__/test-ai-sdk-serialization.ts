import type { LanguageModelV4CallOptions } from '@ai-sdk/provider';
import { temporal } from '@temporalio/proto';
import { helpers } from '@temporalio/test-helpers';
import { AiSdkPlugin } from '..';
import { makeAiSdkTestFunction } from './helpers/ai-sdk';
import type { EmbeddingResponse, ModelResponse } from './stubs/ai-sdk';
import {
  contentResponse,
  embeddingResponse,
  helloWorkflowGenerator,
  TestProvider,
  textResponse,
  toolCallResponse,
} from './stubs/ai-sdk';
import {
  binaryFileWorkflow,
  embeddingWorkflow,
  helloWorldAgent,
  imageToolWorkflow,
} from './workflows/ai-sdk';
import EventType = temporal.api.enums.v1.EventType;

const test = makeAiSdkTestFunction();

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
});

function activityResults(events: temporal.api.history.v1.IHistoryEvent[]): any[] {
  return events
    .filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_COMPLETED)
    .map((e) => {
      const data = e.activityTaskCompletedEventAttributes!.result!.payloads![0]!.data!;
      return JSON.parse(Buffer.from(data).toString());
    });
}

function* bodiesGenerator(prompt: string): Generator<ModelResponse> {
  yield {
    ...textResponse('Test Haiku'),
    request: { body: { messages: [{ role: 'user', content: prompt }] } },
    response: {
      id: 'resp-1',
      modelId: 'gpt-4o-mini',
      timestamp: new Date('2026-01-01T00:00:00Z'),
      headers: { 'x-request-id': 'req-1' },
      body: { choices: [{ message: { content: 'Test Haiku' } }] },
    },
  };
}

test('invokeModel result in history has no request or response body', async (t) => {
  const prompt = 'Tell me about recursion in programming.';
  const { createWorker, startWorkflow } = helpers(t);
  const worker = await createWorker({
    plugins: [new AiSdkPlugin({ modelProvider: new TestProvider(bodiesGenerator(prompt)) })],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(helloWorldAgent, { args: [prompt] });
    t.is(await handle.result(), 'Test Haiku');
    const { events } = await handle.fetchHistory();
    const [result] = activityResults(events!);

    t.deepEqual(result.request, {});
    t.deepEqual(result.response, {
      id: 'resp-1',
      modelId: 'gpt-4o-mini',
      timestamp: '2026-01-01T00:00:00.000Z',
      headers: { 'x-request-id': 'req-1' },
    });
  });
});

function* embeddingBodyGenerator(): Generator<EmbeddingResponse> {
  yield {
    ...embeddingResponse(['Hello, world!', 'How are you?']),
    response: { headers: { 'x-request-id': 'req-1' }, body: { data: 'raw embeddings' } },
  };
}

test('invokeEmbeddingModel result in history has no response body', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const worker = await createWorker({
    plugins: [
      new AiSdkPlugin({
        modelProvider: new TestProvider(helloWorkflowGenerator(), embeddingBodyGenerator()),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(embeddingWorkflow, {
      args: [['Hello, world!', 'How are you?']],
    });
    t.is((await handle.result()).count, 2);
    const { events } = await handle.fetchHistory();
    const results = activityResults(events!);

    // Without `maxEmbeddingsPerCall`, `embedMany` sends all values to one Activity.
    t.is(results.length, 1);
    t.deepEqual(results[0].response, { headers: { 'x-request-id': 'req-1' } });
    t.is(results[0].embeddings.length, 2);
  });
});
