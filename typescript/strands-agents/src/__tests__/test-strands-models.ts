import type { BaseModelConfig, ModelStreamEvent, StreamOptions } from '@strands-agents/sdk';
import { Message, TextBlock } from '@strands-agents/sdk';
import type { TestFn } from 'ava';
import { ApplicationFailure } from '@temporalio/common';
import type { BaseContext } from '@temporalio/test-helpers';
import { test as anyTest, helpers } from '@temporalio/test-helpers';
import { WorkflowStreamClient } from '@temporalio/workflow-streams/client';
import { StrandsPlugin } from '..';
import { heartbeatTimeouts, setupTestEnv, StubModel, textTurn } from './helpers/strands';
import { compressionAgent, defaultModelAgent, helloAgent, retryConfigAgent, streamingAgent } from './workflows/strands';

const test = anyTest as TestFn<BaseContext>;

setupTestEnv(test);

test('TemporalAgent dispatches model.stream() through invokeModel activity', async (t) => {
  t.timeout(60_000);
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new StrandsPlugin({
        models: { test: () => new StubModel([textTurn('hello from stub')]) },
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(helloAgent, {
      args: ['say hi'],
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(await handle.result(), 'hello from stub');
    // The model call heartbeats by default. The config read runs synchronously, so it can't
    // heartbeat and gets no heartbeat timeout.
    const { events } = await handle.fetchHistory();
    t.deepEqual(heartbeatTimeouts(events), [
      ['strandsGetModelConfig', 0],
      ['invokeModel', 10_000],
    ]);
  });
});

test('TemporalAgent without `model:` resolves to the single registered factory', async (t) => {
  t.timeout(60_000);
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new StrandsPlugin({
        models: { onlyModel: () => new StubModel([textTurn('reply from default')]) },
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(defaultModelAgent, {
      args: ['hi'],
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(result, 'reply from default');
  });
});

test('TemporalAgent gives Strands the worker model id and context window', async (t) => {
  t.timeout(60_000);
  const { createWorker, executeWorkflow } = helpers(t);

  const userTurns: string[][] = [];
  class WindowStub extends StubModel {
    // The API key must not reach the workflow, or its history.
    override getConfig(): BaseModelConfig & { apiKey: string } {
      return { modelId: 'window-stub', contextWindowLimit: 1000, apiKey: 'secret-api-key' };
    }
    override async *stream(messages: Message[], options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
      userTurns.push(messages.filter((m) => m.role === 'user').map((m) => (m.content[0] as TextBlock).text));
      yield* super.stream(messages, options);
    }
  }
  // The first reply reports 800 input tokens: 80% of the model's 1,000-token window, over the
  // default compression threshold of 70%, but only 0.4% of Strands' 200,000-token fallback.
  const firstTurn = [
    ...textTurn('first'),
    { type: 'modelMetadataEvent', usage: { inputTokens: 800, outputTokens: 1, totalTokens: 801 } },
  ];

  const worker = await createWorker({
    plugins: [
      new StrandsPlugin({
        models: { test: () => new WindowStub([firstTurn as ModelStreamEvent[], textTurn('second')]) },
      }),
    ],
  });

  const config = await worker.runUntil(
    executeWorkflow(compressionAgent, { args: [['turn 1', 'turn 2']], workflowExecutionTimeout: '30 seconds' })
  );
  // Compression dropped the first turn before the second model call, as it does natively.
  t.deepEqual(
    { config, userTurns },
    { config: { modelId: 'window-stub', contextWindowLimit: 1000 }, userTurns: [['turn 1'], ['turn 2']] }
  );
});

test('TemporalAgent reads the model config again after a failed read', async (t) => {
  t.timeout(60_000);
  const { createWorker, executeWorkflow } = helpers(t);

  class FlakyConfigStub extends StubModel {
    private reads = 0;
    override getConfig(): BaseModelConfig {
      if (++this.reads === 1) throw ApplicationFailure.nonRetryable('config unavailable');
      return { modelId: 'flaky-stub', contextWindowLimit: 1000 };
    }
  }

  const worker = await createWorker({
    plugins: [new StrandsPlugin({ models: { test: () => new FlakyConfigStub([textTurn('second')]) } })],
  });

  const result = await worker.runUntil(executeWorkflow(retryConfigAgent, { args: ['hi'] }));
  t.deepEqual(result, {
    turns: ['error: config unavailable', 'second'],
    config: { modelId: 'flaky-stub', contextWindowLimit: 1000 },
  });
});

test('invokeModel gives the model Message instances and an AbortSignal', async (t) => {
  // Strands content-block and Message classes have lossy `toJSON()` methods.
  // After Temporal's default payload converter serializes activity input,
  // the receiving side sees plain Bedrock-shape objects without the `type`
  // discriminator. ModelActivity must call `Message.fromMessageData` so the
  // user's `Model.stream` receives real Strands types. The agent's
  // `cancelSignal` arrives as a plain object too, and model clients such as
  // OpenAI's call its `addEventListener`.
  t.timeout(60_000);
  const { createWorker, executeWorkflow } = helpers(t);

  // Capture what arrives at stream() so the assertion runs even though the
  // model lives on the worker — record into a module-level sink that's also
  // reachable from the assertion below.
  let receivedFirstMessage: unknown;
  let firstBlockIsTextBlock = false;
  let receivedSignal: unknown;
  class AssertingStub extends StubModel {
    override async *stream(messages: Message[], options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
      receivedSignal = options?.cancelSignal;
      if (receivedFirstMessage === undefined) {
        receivedFirstMessage = messages[0];
        const first = messages[0];
        firstBlockIsTextBlock = first instanceof Message && first.content[0] instanceof TextBlock;
      }
      yield* super.stream(messages, options);
    }
  }

  const worker = await createWorker({
    plugins: [
      new StrandsPlugin({
        models: { test: () => new AssertingStub([textTurn('ok')]) },
      }),
    ],
  });

  await worker.runUntil(async () => {
    await executeWorkflow(helloAgent, { args: ['say hi'] });
  });

  t.true(receivedFirstMessage instanceof Message, 'expected first message to be a Message instance');
  t.true(firstBlockIsTextBlock, 'expected first content block to be a TextBlock instance');
  t.true(receivedSignal instanceof AbortSignal, 'expected cancelSignal to be an AbortSignal');
});

test('streamingTopic publishes model events to a WorkflowStream subscriber', async (t) => {
  const { createWorker, startWorkflow, taskQueue } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new StrandsPlugin({
        models: { test: () => new StubModel([textTurn('streamed')]) },
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(streamingAgent, {
      args: ['say something'],
      workflowExecutionTimeout: '30 seconds',
    });

    const stream = WorkflowStreamClient.create(t.context.env.client, handle.workflowId);
    const events: Array<{ type?: string }> = [];
    const subscriber = (async () => {
      // textTurn yields 5 model events (messageStart, contentBlockStart,
      // contentBlockDelta, contentBlockStop, messageStop); break once we've
      // observed the terminal `modelMessageStopEvent`.
      for await (const item of stream.subscribe<{ type?: string }>(['model'], 0, {
        pollCooldown: '50 milliseconds',
        resultType: true,
      })) {
        events.push(item.data);
        if (item.data.type === 'modelMessageStopEvent') break;
      }
    })();

    t.is(await handle.result(), 'streamed');
    await subscriber;
    t.true(
      events.some((e) => e.type === 'modelMessageStartEvent'),
      'expected a modelMessageStartEvent in the stream'
    );
    t.true(
      events.some((e) => e.type === 'modelMessageStopEvent'),
      'expected a modelMessageStopEvent in the stream'
    );
  });
  // Reference `taskQueue` to keep the helper destructure from being flagged as unused.
  t.truthy(taskQueue);
});
