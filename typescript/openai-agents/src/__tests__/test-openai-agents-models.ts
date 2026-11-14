import { WorkflowClient } from '@temporalio/client';
import { temporal } from '@temporalio/proto';
import { helpers } from '@temporalio/test-helpers';
import { OpenAIAgentsPlugin, OpenAIAgentsTraceClientInterceptor } from '..';
import {
  runModelOverrideWorkflow,
  summaryAgentWorkflow,
  localActivityAgentWorkflow,
  summaryOverrideStringWorkflow,
} from './workflows/openai-agents';
import { FakeModelProvider, textResponse, toolCallResponse, type FakeModel } from './stubs/openai-agents';
import { handoffResponse } from './stubs/openai-agents-fakes';
import { makeOpenAIAgentsTestFunction, replay } from './helpers/openai-agents';
import EventType = temporal.api.enums.v1.EventType;

const test = makeOpenAIAgentsTestFunction();

// Compile-time check: plugin + client-interceptor modelParams reject function/object summary.
{
  const _provider = null! as FakeModelProvider;
  // @ts-expect-error — modelParams.summary must be string, not function/object
  new OpenAIAgentsPlugin({ modelProvider: _provider, modelParams: { summary: { provide: () => 'x' } } });
  // @ts-expect-error — modelParams.summary must be string, not function/object
  new OpenAIAgentsTraceClientInterceptor({ modelParams: { summary: { provide: () => 'x' } } });
}

test('Local activity mode uses local activities for model calls', async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Local activity response')]),
        modelParams: { useLocalActivity: true, startToCloseTimeout: '60s' },
      }),
    ],
  });

  // Interceptor on the start-side wfClient is what injects the config header.
  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
    interceptors: [
      new OpenAIAgentsTraceClientInterceptor({
        modelParams: { useLocalActivity: true, startToCloseTimeout: '60s' },
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await wfClient.start(localActivityAgentWorkflow, {
      taskQueue,
      workflowId: `local-activity-${Date.now()}`,
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.is(result, 'Local activity response');

    const { events } = await handle.fetchHistory();

    // Local activities show up as MarkerRecorded ("core_local_activity"), not ActivityTaskScheduled.
    const markerEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_MARKER_RECORDED) ?? [];
    t.true(
      markerEvents.length > 0,
      `Expected MarkerRecorded events for local activities in history, got ${markerEvents.length}`
    );

    const activityScheduledEvents =
      events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED) ?? [];
    const modelActivities = activityScheduledEvents.filter(
      (e) => e.activityTaskScheduledEventAttributes?.activityType?.name === 'invokeModelActivity'
    );
    t.is(modelActivities.length, 0, `Expected no regular invokeModelActivity, got ${modelActivities.length}`);
  });
});

test('summary string is passed through to model activity', async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Summary test response')]),
        modelParams: { summary: 'Custom model summary' },
      }),
    ],
  });

  // Interceptor on the start-side wfClient is what injects the config header.
  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
    interceptors: [
      new OpenAIAgentsTraceClientInterceptor({
        modelParams: { summary: 'Custom model summary' },
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await wfClient.start(summaryOverrideStringWorkflow, {
      taskQueue,
      workflowId: `summary-override-${Date.now()}`,
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.is(result, 'Summary test response');

    const { events } = await handle.fetchHistory();
    const activityScheduledEvents =
      events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED) ?? [];
    t.true(
      activityScheduledEvents.length >= 1,
      `Expected at least 1 activity scheduled event, got ${activityScheduledEvents.length}`
    );

    const modelEvent = activityScheduledEvents.find(
      (e) => e?.activityTaskScheduledEventAttributes?.activityType?.name === 'invokeModelActivity'
    );
    t.truthy(modelEvent, 'Expected invokeModelActivity in history');
    const userMetadata = (modelEvent as any)?.userMetadata;
    t.truthy(userMetadata, 'Expected userMetadata on activity scheduled event');
    if (userMetadata) {
      const summaryPayload = userMetadata?.summary;
      t.truthy(summaryPayload, 'Expected summary payload in userMetadata');
      if (summaryPayload) {
        const summaryText = Buffer.from(summaryPayload.data).toString('utf-8');
        t.true(
          summaryText.includes('Custom model summary'),
          `Expected summary metadata to contain 'Custom model summary', got: ${summaryText}`
        );
      }
    }
  });
});

test('A ModelSummaryProvider gets the agent of each model call, in nested and resumed runs too', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const modelProvider = new FakeModelProvider([
    toolCallResponse('ask_expert', { input: 'q' }),
    textResponse('expert answer'),
    handoffResponse('transfer_to_billing'),
    toolCallResponse('refund', {}),
    textResponse('billing done'),
  ]);
  const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider })] });
  await worker.runUntil(async () => {
    const handle = await startWorkflow(summaryAgentWorkflow, { workflowExecutionTimeout: '30 seconds' });
    t.is(await handle.result(), 'billing done');
    const { events } = await handle.fetchHistory();
    const summaries = (events ?? [])
      .filter((e) => e.activityTaskScheduledEventAttributes?.activityType?.name === 'invokeModelActivity')
      .map((e) =>
        Buffer.from(e.userMetadata?.summary?.data ?? [])
          .toString('utf-8')
          .replace(/"/g, '')
      );
    t.deepEqual(summaries, ['agent:lead', 'agent:expert', 'agent:lead', 'agent:billing', 'agent:billing']);
    await replay(t, handle);
  });
});

test('runConfig.model leaves an agent that sets its own model on that model, as upstream', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);
  const names: string[] = [];
  const provider = {
    getModel(name?: string) {
      return {
        async getResponse() {
          names.push(name ?? '');
          return textResponse('ok');
        },
      } as unknown as FakeModel;
    },
  };
  const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })] });
  await worker.runUntil(executeWorkflow(runModelOverrideWorkflow, { workflowExecutionTimeout: '30 seconds' }));
  t.deepEqual(names, ['own-model']);
});
