import path from 'node:path';
import type { JSONValue, Message, ModelStreamEvent, StreamOptions } from '@strands-agents/sdk';
import type { TestFn } from 'ava';
import type { temporal } from '@temporalio/proto';
import type { BaseContext } from '@temporalio/test-helpers';
import { test as anyTest, helpers, loadHistory } from '@temporalio/test-helpers';
import { Worker } from '@temporalio/worker';
import { StrandsPlugin } from '..';
import { ModelActivity } from '../model-activity';
import { setupTestEnv, StubModel, textTurn } from './helpers/strands';
import { statefulAgent, statefulChatAgent } from './workflows/strands';

const test = anyTest as TestFn<BaseContext>;

setupTestEnv(test);

/** What each call of a {@link StatefulStub} got: the response id it chained to, and its messages. */
type StatefulCall = { previousResponseId: JSONValue | undefined; messages: number };

/** Keeps the conversation "server-side", as `OpenAIModel` with `stateful: true` does. */
class StatefulStub extends StubModel {
  constructor(
    turns: ModelStreamEvent[][],
    private readonly calls: StatefulCall[]
  ) {
    super(turns);
  }

  override get stateful(): boolean {
    return true;
  }

  override async *stream(messages: Message[], options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
    this.calls.push({ previousResponseId: options?.modelState?.get('responseId'), messages: messages.length });
    options?.modelState?.set('responseId', `resp_${this.calls.length}`);
    yield* super.stream(messages, options);
  }
}

const CHAINED_CALLS = [
  { previousResponseId: undefined, messages: 1 },
  { previousResponseId: 'resp_1', messages: 1 },
];

function scheduledActivities(history: temporal.api.history.v1.IHistory): string[] {
  return (history.events ?? []).flatMap((e) => e.activityTaskScheduledEventAttributes?.activityType?.name ?? []);
}

test('a stateful run recorded before the stateful flag replays on the stateless path', async (t) => {
  // Recorded with `statefulAgent` on 3af00d9, whose config activity returns no `stateful` flag: the
  // run scheduled `invokeModel`, where a config with the flag schedules `strandsInvokeStatefulModel`.
  const history = await loadHistory(path.resolve(__dirname, '../../src/__tests__/histories/stateful-before-flag.json'));
  await t.notThrowsAsync(
    Worker.runReplayHistory(
      {
        workflowBundle: t.context.workflowBundle,
        plugins: [new StrandsPlugin({ models: { stateful: () => new StatefulStub([], []) } })],
      },
      history
    )
  );
});

test('a stateful worker model keeps the conversation server-side', async (t) => {
  t.timeout(60_000);
  const { createWorker, executeWorkflow, startWorkflow } = helpers(t);
  const calls: StatefulCall[] = [];
  const turns = [textTurn('one'), textTurn('two'), textTurn('three'), textTurn('four')];
  const plugin = (): StrandsPlugin => new StrandsPlugin({ models: { stateful: () => new StatefulStub(turns, calls) } });

  const worker = await createWorker({ plugins: [plugin()], maxCachedWorkflows: 0 });
  await worker.runUntil(async () => {
    for (const [mode, activityType] of [
      ['plain', 'strandsInvokeStatefulModel'],
      ['streaming', 'strandsInvokeStatefulModelStreaming'],
    ] as const) {
      calls.length = 0;
      const handle = await startWorkflow(statefulAgent, { args: [mode] });
      t.deepEqual(await handle.result(), [0, 0]);
      t.deepEqual(calls, CHAINED_CALLS);
      const history = await handle.fetchHistory();
      t.deepEqual(scheduledActivities(history), ['strandsGetModelConfig', activityType, activityType]);
      await Worker.runReplayHistory({ workflowBundle: t.context.workflowBundle, plugins: [plugin()] }, history);
    }

    t.is(
      await executeWorkflow(statefulAgent, { args: ['manager'] }),
      'contextManager and conversationManager cannot be used with a stateful model. ' +
        'The model manages conversation state server-side.'
    );
  });
});

test('a stateful model call waits for a worker with stateful support', async (t) => {
  t.timeout(60_000);
  const { createWorker, startWorkflow, taskQueue } = helpers(t);
  const calls: StatefulCall[] = [];
  const plugin = new StrandsPlugin({
    models: { stateful: () => new StatefulStub([textTurn('one'), textTurn('two')], calls) },
  });
  // An activity worker that registers only the stateless model activities, behind this version's
  // workflow and config. This is not a mixed-version rollout: an earlier workflow or config activity
  // breaks the run, as the README says.
  const legacy = new ModelActivity({ stateful: () => new StatefulStub([], calls) });
  const modelsQueue = `${taskQueue}-models`;

  const worker = await createWorker({ plugins: [plugin] });
  await worker.runUntil(async () => {
    const handle = await startWorkflow(statefulAgent, { args: ['models-queue'] });
    const legacyWorker = await createWorker({
      taskQueue: modelsQueue,
      activities: {
        strandsGetModelConfig: legacy.getModelConfig.bind(legacy),
        invokeModel: legacy.invokeModel,
        invokeModelStreaming: legacy.invokeModelStreaming,
      },
    });
    const pending = await legacyWorker.runUntil(async () => {
      for (let polls = 0; ; polls++) {
        const description = await handle.describe();
        const [activity] = description.raw.pendingActivities ?? [];
        if (activity && (activity.attempt ?? 0) > 2) return activity;
        if (description.status.name !== 'RUNNING' || polls === 300) {
          throw new Error(
            `no retried model activity: run ${description.status.name}, pending ` +
              JSON.stringify(description.raw.pendingActivities)
          );
        }
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
    });
    t.is(pending.activityType?.name, 'strandsInvokeStatefulModel');
    t.regex(pending.lastFailure?.message ?? '', /not registered/);
    t.deepEqual(calls, []);

    const upgradedWorker = await createWorker({ taskQueue: modelsQueue, plugins: [plugin] });
    t.deepEqual(await upgradedWorker.runUntil(handle.result()), [0, 0]);
  });
  t.deepEqual(calls, CHAINED_CALLS);
});

test("a model call fails at once when the worker model's stateful flag differs from the run's config", async (t) => {
  t.timeout(60_000);
  const { createWorker, startWorkflow, taskQueue } = helpers(t);
  const calls: StatefulCall[] = [];
  const plugin = new StrandsPlugin({ models: { stateful: () => new StatefulStub([], calls) } });
  // The worker model now in one mode, as after a model change or an upgrade.
  class SwitchedStub extends StatefulStub {
    constructor(private readonly isStateful: boolean) {
      super([textTurn('one')], calls);
    }
    override get stateful(): boolean {
      return this.isStateful;
    }
  }

  for (const c of [
    // The run's config is stateless (here an earlier version's, with no flag; a model that became
    // stateful later gives the same path), so its call to the stateful model reaches the stateless
    // activity.
    {
      config: {},
      stateful: true,
      activity: 'invokeModel',
      type: 'StatefulModelOnStatelessPath',
      message: /is stateful, .* became stateful after the run read its config/,
    },
    // The config says stateful, but the model became stateless.
    {
      config: { stateful: true },
      stateful: false,
      activity: 'strandsInvokeStatefulModel',
      type: 'StatelessModelOnStatefulPath',
      message: /is not stateful, .* became stateless after the run read its config/,
    },
  ]) {
    const current = new ModelActivity({ stateful: () => new SwitchedStub(c.stateful) });
    const worker = await createWorker({ plugins: [plugin] });
    const history = await worker.runUntil(async () => {
      const handle = await startWorkflow(statefulAgent, { args: ['models-queue'] });
      const modelsWorker = await createWorker({
        taskQueue: `${taskQueue}-models`,
        activities: {
          strandsGetModelConfig: async () => c.config,
          invokeModel: current.invokeModel,
          invokeModelStreaming: current.invokeModelStreaming,
          strandsInvokeStatefulModel: current.invokeStatefulModel,
          strandsInvokeStatefulModelStreaming: current.invokeStatefulModelStreaming,
        },
      });
      await modelsWorker.runUntil(handle.result());
      return handle.fetchHistory();
    });

    t.deepEqual(scheduledActivities(history), ['strandsGetModelConfig', c.activity], c.type);
    const events = history.events ?? [];
    const attempts = events.flatMap((e) => e.activityTaskStartedEventAttributes?.attempt ?? []);
    const failures = events.flatMap((e) => e.activityTaskFailedEventAttributes?.failure ?? []);
    t.deepEqual(attempts, [1, 1], c.type);
    t.is(failures.length, 1, c.type);
    t.like(failures[0], { applicationFailureInfo: { type: c.type, nonRetryable: true } });
    t.regex(failures[0]?.message ?? '', c.message);
    t.deepEqual(calls, [], c.type);
  }
});

test("the README's continue-as-new handoff carries a stateful model's state", async (t) => {
  t.timeout(60_000);
  const { createWorker, executeWorkflow } = helpers(t);
  const calls: StatefulCall[] = [];
  const worker = await createWorker({
    plugins: [
      new StrandsPlugin({ models: { stateful: () => new StatefulStub([textTurn('one'), textTurn('two')], calls) } }),
    ],
  });

  const messages = await worker.runUntil(
    executeWorkflow(statefulChatAgent, { args: [{ prompts: ['turn one', 'turn two'] }] })
  );
  t.deepEqual({ messages, calls }, { messages: 0, calls: CHAINED_CALLS });
});
