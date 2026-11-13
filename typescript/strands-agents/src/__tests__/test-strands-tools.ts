import type { Message, ModelStreamEvent, StreamOptions, ToolSpec } from '@strands-agents/sdk';
import type { TestFn } from 'ava';
import { ApplicationFailure } from '@temporalio/common';
import type { BaseContext } from '@temporalio/test-helpers';
import { test as anyTest, helpers } from '@temporalio/test-helpers';
import { Worker } from '@temporalio/worker';
import { StrandsPlugin } from '..';
import { auditLog, auditTool, getWeather, getWeatherCalls } from './activities/strands';
import { setupTestEnv, StubModel, textTurn, toolCallTurn } from './helpers/strands';
import {
  activityToolAgent,
  agentToolAgent,
  failingActivityToolAgent,
  hooksAgent,
  inWorkflowToolAgent,
  structuredOutputAgent,
} from './workflows/strands';

const test = anyTest as TestFn<BaseContext>;

setupTestEnv(test);

for (const mode of ['default', 'reset', 'implicit', 'preserve'] as const) {
  test(`agents as tools: ${mode} context behavior and replay`, async (t) => {
    const { createWorker, startWorkflow } = helpers(t);
    const calls: string[][] = [];
    class SpecialistModel extends StubModel {
      override async *stream(messages: Message[], options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
        calls.push(
          messages.map((message) =>
            message.content.map((block) => (block.type === 'textBlock' ? block.text : '')).join('')
          )
        );
        yield* super.stream(messages, options);
      }
    }
    const plugin = (): StrandsPlugin =>
      new StrandsPlugin({
        models: {
          parent: () =>
            new StubModel([
              toolCallTurn('specialist', 'call_1', { input: 'first request' }),
              toolCallTurn('specialist', 'call_2', { input: 'second request' }),
              textTurn('done'),
            ]),
          specialist: () => new SpecialistModel([textTurn('first answer'), textTurn('second answer')]),
        },
      });
    const worker = await createWorker({ plugins: [plugin()], maxCachedWorkflows: 0 });
    const history = await worker.runUntil(async () => {
      const handle = await startWorkflow(agentToolAgent, {
        args: [mode],
        workflowExecutionTimeout: '30 seconds',
      });
      t.deepEqual(await handle.result(), {
        reply: 'done',
        appStates: [{ calls: 0 }, mode === 'preserve' ? { calls: 1, duringCall: 'added' } : { calls: 0 }],
        toolResults: [
          { status: 'success', text: 'first answer' },
          { status: 'success', text: 'second answer' },
        ],
      });
      return handle.fetchHistory();
    });

    const baseline = ['baseline prompt', 'baseline reply'];
    t.deepEqual(calls, [
      [...baseline, 'first request'],
      [...baseline, ...(mode === 'preserve' ? ['first request', 'first answer'] : []), 'second request'],
    ]);
    await Worker.runReplayHistory({ workflowBundle: t.context.workflowBundle, plugins: [plugin()] }, history);
    t.is(calls.length, 2);
  });
}

test('activityAsTool dispatches a registered activity from the agent loop', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new StrandsPlugin({
        models: {
          test: () =>
            new StubModel([
              toolCallTurn('getWeather', 'call_1', { location: 'Tokyo' }),
              textTurn('the weather is sunny'),
            ]),
        },
      }),
    ],
    activities: { getWeather },
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(activityToolAgent, { args: ['weather in Tokyo?'] });
    t.is(result, 'the weather is sunny');
  });

  // Assert the activity was actually invoked with the model's tool input — this
  // would fail if the plugin returned the scripted reply without dispatching it.
  // `getWeatherCalls` is a process-wide sink shared with other tests that ava may
  // run concurrently, so filter to this test's unique location ('Tokyo'); no
  // other test invokes getWeather with it.
  t.deepEqual(
    getWeatherCalls.filter((c) => c.location === 'Tokyo'),
    [{ location: 'Tokyo' }]
  );
});

test('activityAsTool without a description exposes a valid spec and dispatches the registered activity', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);
  let offeredTools: ToolSpec[] | undefined;
  class CapturingStub extends StubModel {
    override async *stream(messages: Message[], options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
      offeredTools ??= options?.toolSpecs;
      yield* super.stream(messages, options);
    }
  }

  const worker = await createWorker({
    plugins: [
      new StrandsPlugin({
        models: {
          test: () =>
            new CapturingStub([
              toolCallTurn('weather', 'call_1', { location: 'Berlin' }),
              textTurn('the weather is sunny'),
            ]),
        },
      }),
    ],
    activities: { getWeather },
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(activityToolAgent, {
      args: ['weather in Berlin?', { name: 'weather' }],
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(result, 'the weather is sunny');
  });

  t.like(offeredTools, [{ name: 'weather' }]);
  t.false(Object.hasOwn(offeredTools![0]!, 'description'));
  t.deepEqual(
    getWeatherCalls.filter((c) => c.location === 'Berlin'),
    [{ location: 'Berlin' }]
  );
});

test('activityAsTool gives the model the error the activity threw', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new StrandsPlugin({
        models: {
          test: () =>
            new StubModel([toolCallTurn('getWeather', 'call_1', { location: 'Atlantis' }), textTurn('no weather')]),
        },
      }),
    ],
    activities: {
      async getWeather(input: { location: string }): Promise<never> {
        throw ApplicationFailure.nonRetryable(`Unknown location: ${input.location}`);
      },
    },
  });

  await worker.runUntil(async () => {
    // Not the ActivityFailure's own "Activity task failed".
    t.deepEqual(await executeWorkflow(failingActivityToolAgent, { args: ['weather in Atlantis?'] }), [
      { text: 'Unknown location: Atlantis', error: 'Unknown location: Atlantis' },
    ]);
  });
});

test('in-workflow tool() runs inside the workflow sandbox', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  // Capture the follow-up model call's messages: the tool's result carries an
  // `echoed:` marker the stub model never emits, so seeing it proves the
  // callback actually ran (not just that the scripted reply was returned).
  let secondCallMessages: Message[] | undefined;
  class CapturingStub extends StubModel {
    private call = 0;
    override async *stream(messages: Message[], options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
      if (this.call++ === 1) secondCallMessages = messages;
      yield* super.stream(messages, options);
    }
  }

  const worker = await createWorker({
    plugins: [
      new StrandsPlugin({
        models: {
          test: () =>
            new CapturingStub([
              toolCallTurn('echo', 'call_1', { text: 'hi from in-workflow tool' }),
              textTurn('echoed'),
            ]),
        },
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(inWorkflowToolAgent, { args: ['say hi'] });
    t.is(result, 'echoed');
  });

  t.true(JSON.stringify(secondCallMessages).includes('echoed: hi from in-workflow tool'));
});

test('activityAsHook dispatches a registered activity on AfterToolCallEvent', async (t) => {
  auditLog.length = 0;
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new StrandsPlugin({
        models: {
          test: () => new StubModel([toolCallTurn('echo', 'call_1', { text: 'hello' }), textTurn('done')]),
        },
      }),
    ],
    activities: { auditTool },
  });

  await worker.runUntil(async () => {
    const fired = await executeWorkflow(hooksAgent, { args: ['say hello'] });
    t.deepEqual(fired, ['echo']);
  });
  t.deepEqual(auditLog, ['echo']);
});

test('TemporalAgent surfaces structured output via structuredOutputSchema', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const person = { name: 'John Smith', age: 30, occupation: 'software engineer' };
  const worker = await createWorker({
    plugins: [
      new StrandsPlugin({
        models: {
          // Strands' structured-output flow exposes a fixed tool named
          // `strands_structured_output`; the model's tool-call turn carries the
          // validated object as the tool input.
          test: () => new StubModel([toolCallTurn('strands_structured_output', 'call_1', person)]),
        },
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(structuredOutputAgent, { args: ['describe John'] });
    t.deepEqual(result, person);
  });
});
