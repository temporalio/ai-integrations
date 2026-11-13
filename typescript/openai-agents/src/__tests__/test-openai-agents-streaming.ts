import { WorkflowClient } from '@temporalio/client';
import { temporal } from '@temporalio/proto';
import { helpers } from '@temporalio/test-helpers';
import { WorkflowStreamClient } from '@temporalio/workflow-streams/client';
import { OpenAIAgentsPlugin, OpenAIAgentsTraceClientInterceptor } from '..';
import {
  streamingAgentWorkflow,
  streamingLocalActivityWorkflow,
  streamingNoTopicWorkflow,
} from './workflows/openai-agents';
import { StreamingFakeModelProvider, streamingTextEvents } from './stubs/openai-agents-fakes';
import { makeOpenAIAgentsTestFunction } from './helpers/openai-agents';
import EventType = temporal.api.enums.v1.EventType;

const test = makeOpenAIAgentsTestFunction();

test('Streaming: external subscriber and in-workflow result see the same events', async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  const events = streamingTextEvents('Hello streamed world');
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new StreamingFakeModelProvider(events),
        modelParams: { streamingTopic: 'events', streamingBatchInterval: '50ms' },
      }),
    ],
  });

  // Match the modelParams the plugin uses so the config header carries streamingTopic.
  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
    interceptors: [
      new OpenAIAgentsTraceClientInterceptor({
        modelParams: { streamingTopic: 'events', streamingBatchInterval: '50ms' },
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await wfClient.start(streamingAgentWorkflow, {
      taskQueue,
      workflowId: `streaming-${Date.now()}`,
      args: ['Hi'],
      workflowExecutionTimeout: '30 seconds',
    });

    const streamClient = WorkflowStreamClient.create(t.context.env.client, handle.workflowId);
    const received: Array<{ type?: string; delta?: string }> = [];
    const gen = streamClient.topic<{ type?: string; delta?: string }>('events').subscribe(0, { pollCooldown: 0 });
    const collect = (async () => {
      for await (const item of gen) {
        received.push(item.data);
        if (received.length >= events.length) {
          await gen.return();
          break;
        }
      }
    })();

    const result = await handle.result();
    await collect;

    t.deepEqual(result.deltas, ['Hello streamed world'], 'in-workflow result yields the text deltas in order');
    t.is(result.finalOutput, 'Hello streamed world');

    t.is(received.length, events.length, 'external subscriber receives every published event in order');
    t.is(received[0]!.type, 'output_text_delta');
    t.is(received[0]!.delta, 'Hello streamed world');
    t.is(received[received.length - 1]!.type, 'response_done');
  });
});

test('Streaming: missing streamingTopic fails fast before scheduling an Activity', async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new StreamingFakeModelProvider(streamingTextEvents('unused')),
      }),
    ],
  });

  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
  });

  await worker.runUntil(async () => {
    const handle = await wfClient.start(streamingNoTopicWorkflow, {
      taskQueue,
      workflowId: `streaming-no-topic-${Date.now()}`,
      args: ['Hi'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.is(result, 'StreamingTopicNotConfigured');

    const { events } = await handle.fetchHistory();
    const modelStreamScheduled =
      events?.filter(
        (e) =>
          e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED &&
          e.activityTaskScheduledEventAttributes?.activityType?.name === 'invokeModelStreamActivity'
      ) ?? [];
    t.is(modelStreamScheduled.length, 0, 'no streaming Activity should be scheduled when the topic is unset');
  });
});

test('Streaming: useLocalActivity fails fast before scheduling an Activity', async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new StreamingFakeModelProvider(streamingTextEvents('unused')),
      }),
    ],
  });

  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
  });

  await worker.runUntil(async () => {
    const handle = await wfClient.start(streamingLocalActivityWorkflow, {
      taskQueue,
      workflowId: `streaming-local-activity-${Date.now()}`,
      args: ['Hi'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.is(result, 'StreamingLocalActivityUnsupported');
  });
});
