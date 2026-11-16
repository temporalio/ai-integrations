import * as fs from 'node:fs';
import * as path from 'node:path';
import { randomUUID } from 'node:crypto';
import { WorkflowClient } from '@temporalio/client';
import { defaultPayloadConverter } from '@temporalio/common';
import { historyFromJSON, historyToJSON } from '@temporalio/common/lib/proto-utils';
import { helpers } from '@temporalio/test-helpers';
import { Worker } from '@temporalio/worker';
import { OpenAIAgentsPlugin, OpenAIAgentsTraceClientInterceptor } from '..';
import { makeOpenAIAgentsTestFunction, replay } from './helpers/openai-agents';
import { findPayloadChange } from './helpers/payload-round-trip';
import { FakeModelProvider, textResponse, toolCallResponse } from './stubs/openai-agents';
import { handoffResponse, StreamingFakeModelProvider, streamingTextEvents } from './stubs/openai-agents-fakes';
import {
  temporalModelConfigurationWorkflow,
  temporalModelConcurrentWorkflow,
  temporalModelNestedWorkflow,
  temporalModelLocalWorkflow,
  temporalModelValidationWorkflow,
  temporalModelStreamingWorkflow,
  temporalModelApprovalWorkflow,
} from './workflows/openai-agents';

const test = makeOpenAIAgentsTestFunction();
const scheduled = (history: Awaited<ReturnType<import('@temporalio/client').WorkflowHandle['fetchHistory']>>) =>
  (history.events ?? []).filter((event) =>
    event.activityTaskScheduledEventAttributes?.activityType?.name?.startsWith('invokeModel')
  );

function input(event: ReturnType<typeof scheduled>[number]): Record<string, any> {
  return defaultPayloadConverter.fromPayload(event.activityTaskScheduledEventAttributes!.input!.payloads![0]!);
}

function summary(event: ReturnType<typeof scheduled>[number]): string | undefined {
  const payload = event.userMetadata?.summary;
  return payload?.data?.length ? defaultPayloadConverter.fromPayload(payload) : undefined;
}

function provider() {
  return new FakeModelProvider(Array.from({ length: 12 }, () => textResponse('done')));
}

const REPLAY_MAJOR = 2;
const fixture = path.resolve(
  __dirname,
  '../../src/__tests__/histories',
  `${REPLAY_MAJOR}.x`,
  'temporal-model-options.json'
);

test.serial(
  'TemporalModel and run Activity options override client defaults per field without changing JS model precedence',
  async (t) => {
    const { createWorker, taskQueue } = helpers(t);
    const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider() })] });
    const client = new WorkflowClient({
      connection: t.context.env.connection,
      interceptors: [
        new OpenAIAgentsTraceClientInterceptor({
          modelParams: {
            startToCloseTimeout: '5m',
            scheduleToCloseTimeout: '30m',
            heartbeatTimeout: '7s',
            retry: { maximumAttempts: 2 },
            summary: 'client',
          },
        }),
      ],
    });
    await worker.runUntil(async () => {
      const handle = await client.start(temporalModelConfigurationWorkflow, { workflowId: randomUUID(), taskQueue });
      t.true(await handle.result(), 'agent identity, cloning, and shared model identity survive the run');
      const history = await handle.fetchHistory();
      const events = scheduled(history);
      t.deepEqual(
        events.map((e) => Number(e.activityTaskScheduledEventAttributes?.startToCloseTimeout?.seconds)),
        [1200, 1200, 40, 90, 300, 300]
      );
      t.deepEqual(
        events.map((e) => Number(e.activityTaskScheduledEventAttributes?.heartbeatTimeout?.seconds)),
        [7, 7, 2, 7, 7, 7]
      );
      t.true(
        events.every((e) => Number(e.activityTaskScheduledEventAttributes?.scheduleToCloseTimeout?.seconds) === 1800)
      );
      t.true(events.every((e) => e.activityTaskScheduledEventAttributes?.retryPolicy?.maximumAttempts === 2));
      t.deepEqual(
        events.map((e) => input(e).modelName),
        ['own-model', 'own-model', 'own-model', 'fallback-model', 'string-model', '']
      );
      t.deepEqual(events.map(summary), ['model:Owned', 'model:Clone', 'run', 'fallback', 'client', 'client']);
      for (const event of events) t.is(findPayloadChange(input(event)), undefined);
      await replay(t, handle);
      if (process.env.UPDATE_TEMPORAL_MODEL_HISTORY === '1') fs.writeFileSync(fixture, historyToJSON(history) + '\n');
    });
  }
);

test.serial('TemporalModel Activity options have a committed replay history', async (t) => {
  await t.notThrowsAsync(
    Worker.runReplayHistory(
      {
        workflowBundle: t.context.workflowBundle,
        plugins: [new OpenAIAgentsPlugin({ modelProvider: new FakeModelProvider([]) })],
      },
      historyFromJSON(JSON.parse(fs.readFileSync(fixture, 'utf8')))
    )
  );
});

test('concurrent runs sharing a TemporalModel retain their own Activity options and summary agent', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider() })] });
  await worker.runUntil(async () => {
    const handle = await startWorkflow(temporalModelConcurrentWorkflow, {});
    t.deepEqual(await handle.result(), ['done', 'done']);
    const events = scheduled(await handle.fetchHistory());
    t.true(events.every((e) => input(e).modelName === ''));
    t.deepEqual(
      events.map((e) => input(e).request.modelSettings),
      [{}, {}]
    );
    t.deepEqual(
      events
        .map((e) => [Number(e.activityTaskScheduledEventAttributes?.startToCloseTimeout?.seconds), summary(e)])
        .sort((a, b) => Number(a[0]) - Number(b[0])),
      [
        [11, 'model:Agent11'],
        [22, 'model:Agent22'],
      ]
    );
    await replay(t, handle);
  });
});

for (const kind of ['handoff', 'tool', 'override', 'fresh', 'fallback-tool'] as const) {
  test(`TemporalModel options follow ${kind} calls and inherited run overrides`, async (t) => {
    const { createWorker, startWorkflow } = helpers(t);
    const responses = [
      kind === 'handoff' ? handoffResponse('transfer_to_Child') : toolCallResponse('delegate', { input: 'Hello' }),
      textResponse('done'),
    ];
    if (kind !== 'handoff') responses.push(textResponse('done'));
    const worker = await createWorker({
      plugins: [new OpenAIAgentsPlugin({ modelProvider: new FakeModelProvider(responses) })],
    });
    await worker.runUntil(async () => {
      const handle = await startWorkflow(temporalModelNestedWorkflow, { args: [kind] });
      t.is(await handle.result(), 'done');
      const events = scheduled(await handle.fetchHistory());
      t.deepEqual(
        events.map((e) => Number(e.activityTaskScheduledEventAttributes?.startToCloseTimeout?.seconds)),
        kind === 'handoff'
          ? [60, 33]
          : kind === 'override'
            ? [44, 44, 44]
            : kind === 'fresh'
              ? [44, 33, 44]
              : kind === 'fallback-tool'
                ? [60, 90, 60]
                : [60, 33, 60]
      );
      t.deepEqual(
        events.map((e) => input(e).modelName),
        kind === 'handoff'
          ? ['parent', 'child']
          : kind === 'fallback-tool'
            ? ['parent', 'nested-fallback', 'parent']
            : ['parent', 'child', 'parent']
      );
      t.deepEqual(
        events.map(summary),
        kind === 'handoff'
          ? ['Parent', 'Child']
          : kind === 'fallback-tool'
            ? ['Parent', undefined, 'Parent']
            : ['Parent', 'Child', 'Parent']
      );
      await replay(t, handle);
    });
  });
}

test('TemporalModel and run options can select local model Activities', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider() })] });
  await worker.runUntil(async () => {
    const handle = await startWorkflow(temporalModelLocalWorkflow, {});
    t.is(await handle.result(), 'done');
    const history = await handle.fetchHistory();
    t.is(scheduled(history).length, 0);
    t.is(
      history.events?.filter((e) => e.markerRecordedEventAttributes?.markerName === 'core_local_activity').length,
      2
    );
    await replay(t, handle);
  });
});

test('effective local and streaming options are refused before scheduling a model Activity', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider() })] });
  await worker.runUntil(async () => {
    const handle = await startWorkflow(temporalModelValidationWorkflow, {});
    t.deepEqual(await handle.result(), [
      'UnsupportedLocalModelActivityOptions',
      'UnsupportedLocalModelActivityOptions',
      'StreamingTopicNotConfigured',
      'StreamingLocalActivityUnsupported',
    ]);
    t.is(scheduled(await handle.fetchHistory()).length, 0);
    await replay(t, handle);
  });
});

for (const kind of ['model', 'run', 'fallback'] as const) {
  test(`streaming uses ${kind} Activity options above the client's local default`, async (t) => {
    const { createWorker, taskQueue } = helpers(t);
    const worker = await createWorker({
      plugins: [
        new OpenAIAgentsPlugin({ modelProvider: new StreamingFakeModelProvider(streamingTextEvents('streamed')) }),
      ],
    });
    const client = new WorkflowClient({
      connection: t.context.env.connection,
      interceptors: [new OpenAIAgentsTraceClientInterceptor({ modelParams: { useLocalActivity: true } })],
    });
    await worker.runUntil(async () => {
      const handle = await client.start(temporalModelStreamingWorkflow, {
        workflowId: randomUUID(),
        taskQueue,
        args: [kind],
      });
      t.is(await handle.result(), 'streamed');
      const events = scheduled(await handle.fetchHistory());
      t.is(events.length, 1);
      t.is(Number(events[0]!.activityTaskScheduledEventAttributes?.startToCloseTimeout?.seconds), 21);
      t.is(input(events[0]!).streamingTopic, 'events');
      t.is(input(events[0]!).streamingBatchInterval, '123ms');
      await replay(t, handle);
    });
  });
}

test('a TemporalModel survives approval state serialization and resumes with new run Activity options', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([toolCallResponse('confirm', {}), textResponse('done')]),
      }),
    ],
  });
  await worker.runUntil(async () => {
    const handle = await startWorkflow(temporalModelApprovalWorkflow, {});
    t.is(await handle.result(), 'done');
    const events = scheduled(await handle.fetchHistory());
    t.deepEqual(
      events.map((e) => Number(e.activityTaskScheduledEventAttributes?.startToCloseTimeout?.seconds)),
      [33, 44]
    );
    t.deepEqual(
      events.map((e) => input(e).modelName),
      ['approval', 'approval']
    );
    t.deepEqual(events.map(summary), ['Approval', 'Approval']);
    await replay(t, handle);
  });
});
