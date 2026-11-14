import anyTest from 'ava';
import type { TestFn } from 'ava';
import type {
  EmbeddingModelV4,
  EmbeddingModelV4Result,
  ImageModelV4,
  LanguageModelV4,
  LanguageModelV4GenerateResult,
  ProviderV4,
} from '@ai-sdk/provider';
import { MockActivityEnvironment, TestWorkflowEnvironment } from '@temporalio/testing';
import { Worker } from '@temporalio/worker';
import { AiSdkPlugin, createActivities } from '..';
import type { InvokeEmbeddingModelArgs, InvokeModelArgs, InvokeModelResult } from '..';
import { startLocalWithRetry } from './helpers/environment';
import { findPayloadChange } from './helpers/payload-round-trip';
import { payloadContractWorkflow } from './workflows/payload-contract';

const test = anyTest as TestFn<{ env: TestWorkflowEnvironment }>;

const timestamp = new Date('2026-01-01T00:00:00Z');
const bytes = new Uint8Array([0, 1, 2, 254, 255]);

function generateResult(): LanguageModelV4GenerateResult {
  return {
    content: [
      { type: 'text', text: 'done' },
      { type: 'file', mediaType: 'application/octet-stream', data: { type: 'data', data: bytes } },
      { type: 'reasoning-file', mediaType: 'image/png', data: { type: 'data', data: bytes } },
    ],
    finishReason: { unified: 'stop', raw: 'stop' },
    usage: {
      inputTokens: { total: 1, noCache: 1, cacheRead: undefined, cacheWrite: undefined },
      outputTokens: { total: 1, text: 1, reasoning: undefined },
    },
    warnings: [],
    request: { body: { messages: [{ role: 'user', content: 'hi' }] } },
    response: {
      id: 'resp-1',
      modelId: 'm',
      timestamp,
      headers: { 'x-request-id': 'req-1' },
      body: new Uint8Array([123, 125]),
    },
  };
}

function embedResult(): EmbeddingModelV4Result {
  return {
    embeddings: [[0.5]],
    usage: { tokens: 1 },
    warnings: [],
    response: { headers: { 'x-request-id': 'req-1' }, body: new Uint8Array([123, 125]) },
  };
}

const provider: ProviderV4 = {
  specificationVersion: 'v4',
  languageModel: (): LanguageModelV4 => ({
    specificationVersion: 'v4',
    provider: 'contract-test',
    modelId: 'm',
    supportedUrls: {},
    doGenerate: async () => generateResult(),
    doStream: async () => {
      throw new Error('Not implemented.');
    },
  }),
  embeddingModel: (): EmbeddingModelV4 => ({
    specificationVersion: 'v4',
    provider: 'contract-test',
    modelId: 'e',
    maxEmbeddingsPerCall: undefined,
    supportsParallelCalls: true,
    doEmbed: async () => embedResult(),
  }),
  imageModel: (): ImageModelV4 => {
    throw new Error('Not implemented.');
  },
};

type Activities = {
  invokeModel(args: InvokeModelArgs): Promise<InvokeModelResult>;
  invokeEmbeddingModel(args: InvokeEmbeddingModelArgs): Promise<EmbeddingModelV4Result>;
};

test.before(async (t) => {
  t.context.env = await startLocalWithRetry(() => TestWorkflowEnvironment.createLocal());
});

test.after.always(async (t) => {
  await t.context.env?.teardown();
});

test('invokeModel result survives the converter except the timestamp', async (t) => {
  const activities = createActivities(provider) as Activities;
  const args: InvokeModelArgs = {
    modelId: 'm',
    options: { prompt: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }] },
  };
  const env = new MockActivityEnvironment();
  const result: InvokeModelResult = await env.run(activities.invokeModel, args);

  t.deepEqual(findPayloadChange(result), {
    path: '$.response.timestamp',
    sent: 'Date',
    received: JSON.stringify(timestamp.toISOString()),
  });
  // `TemporalLanguageModel.doGenerate` turns the string back into a `Date`, which the
  // Workflow test below checks. Everything else must survive as it is.
  const response = { ...result.response, timestamp: timestamp.toISOString() };
  t.is(findPayloadChange({ ...result, response }), undefined);
});

test('invokeEmbeddingModel result survives the converter', async (t) => {
  const activities = createActivities(provider) as Activities;
  const result = await new MockActivityEnvironment().run(activities.invokeEmbeddingModel, {
    modelId: 'e',
    values: ['a'],
  });

  t.is(findPayloadChange(result), undefined);
});

async function runContractWorkflow(env: TestWorkflowEnvironment, withUrl: boolean) {
  const taskQueue = `payload-contract-${withUrl}`;
  const worker = await Worker.create({
    connection: env.nativeConnection,
    namespace: env.namespace,
    taskQueue,
    workflowsPath: require.resolve('./workflows/payload-contract'),
    interceptors: { workflowModules: [require.resolve('./workflows/payload-interceptors')] },
    plugins: [new AiSdkPlugin({ modelProvider: provider })],
  });
  return await worker.runUntil(
    env.client.workflow.execute(payloadContractWorkflow, {
      args: [withUrl],
      taskQueue,
      workflowId: `${taskQueue}-${Date.now()}`,
    })
  );
}

test('Workflow sends model Activity input that survives the converter', async (t) => {
  t.is(await runContractWorkflow(t.context.env, false), 'timestamp is Date: true');
});

test('Workflow sends a prompt file URL as a string for the Activity to restore', async (t) => {
  // `prepareFileUrls` in the model Activities turns the string back into a `URL`.
  t.is(
    await runContractWorkflow(t.context.env, true),
    'error: $[0].options.prompt[0].content[2].data.url: URL became "https://example.test/a.png"'
  );
});
