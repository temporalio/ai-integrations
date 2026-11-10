import test from 'ava';
import { WorkflowFailedError } from '@temporalio/client';
import { LangGraphPlugin } from '../index';
import { executionCounts, registry, resetExecutionCounts } from './graphs';
import { execute, replay, setupTestEnv, uid, withWorker } from './helpers';
import {
  continueAsNewFunctionalWorkflow,
  interruptFunctionalWorkflow,
  partialExecutionWorkflow,
  runEntrypoint,
} from './workflows';

const env = setupTestEnv(test);

// Some tests count task executions in this process, so they run serially.

test.serial('a task reads the run context', async (t) => {
  const plugin = new LangGraphPlugin(registry.contextFunctional!);
  const taskQueue = uid('functional-context');
  const cases = [
    { context: { userId: 'alice' }, userId: 'alice', contextType: 'object' },
    { context: { userId: 'bob', tenant: { name: 'acme' } }, userId: 'bob:acme', contextType: 'object' },
    { context: undefined, userId: '', contextType: 'none' },
  ];
  const workflowIds = cases.map(() => uid('functional-context'));
  const results = await withWorker(env(), { taskQueue, plugin }, () =>
    Promise.all(
      cases.map(({ context }, i) =>
        execute(env(), runEntrypoint, {
          taskQueue,
          workflowId: workflowIds[i],
          args: ['contextFunctional', 'context', 10, { context }],
        })
      )
    )
  );
  t.deepEqual(
    results,
    cases.map(({ userId, contextType }) => ({ value: 10, user_id: userId, context_type: contextType }))
  );
  for (const workflowId of workflowIds) await replay(env(), plugin, workflowId);
});

const typedValues = { message: 'AIMessage', when: 'Date', data: 'Uint8Array', nested: 'AIMessage' };

for (const [name, key, expected] of [
  ['activity tasks', 'simpleFunctional', 30],
  ['activity and workflow tasks', 'simpleFunctionalMixed', 35],
  ['typed values', 'typedValuesFunctional', [typedValues, typedValues]],
] as const) {
  test.serial(`a simple entrypoint: ${name}`, async (t) => {
    const taskQueue = uid('e2e-functional');
    const result = await withWorker(env(), { taskQueue, plugin: new LangGraphPlugin(registry[key]!) }, () =>
      execute(env(), runEntrypoint, { taskQueue, args: [key, 'e2e_simple_functional', 10] })
    );
    t.deepEqual(result.result, expected);
  });
}

test.serial('each task executes once across continue-as-new', async (t) => {
  resetExecutionCounts();
  const taskQueue = uid('e2e-continue-as-new');
  // 10 * 3 = 30 -> + 100 = 130 -> * 2 = 260
  const result = await withWorker(env(), { taskQueue, plugin: new LangGraphPlugin(registry.continueAsNewFunctional!) }, () =>
    execute(env(), continueAsNewFunctionalWorkflow, { taskQueue, args: [{ value: 10 }] })
  );
  t.deepEqual(result, { result: 260, task_b_type: 'AIMessage' });
  t.deepEqual(executionCounts, { task_a: 1, task_b: 1, task_c: 1 });
});

test.serial('the cached tasks of a partial run are not executed again', async (t) => {
  resetExecutionCounts();
  const taskQueue = uid('e2e-partial');
  // 10 * 2 = 20 -> + 5 = 25 -> * 3 = 75 -> - 10 = 65 -> + 100 = 165
  const result = await withWorker(env(), { taskQueue, plugin: new LangGraphPlugin(registry.partialExecution!) }, () =>
    execute(env(), partialExecutionWorkflow, { taskQueue, args: [{ value: 10 }] })
  );
  t.deepEqual(result, { result: 165, completed_tasks: 5 });
  t.deepEqual(executionCounts, { step_1: 1, step_2: 1, step_3: 1, step_4: 1, step_5: 1 });
});

test.serial('a task interrupts and resumes', async (t) => {
  const taskQueue = uid('v2-interrupt');
  const result = await withWorker(env(), { taskQueue, plugin: new LangGraphPlugin(registry.interruptFunctional!) }, () =>
    execute(env(), interruptFunctionalWorkflow, { taskQueue, args: ['hello'] })
  );
  t.deepEqual(result, { input: 'hello', answer: 'approved' });
});

test.serial("a task's activityOptions override defaultActivityOptions", async (t) => {
  const taskQueue = uid('e2e-per-task-options');
  await withWorker(env(), { taskQueue, plugin: new LangGraphPlugin(registry.slowFunctional!) }, async () => {
    await t.throwsAsync(execute(env(), runEntrypoint, { taskQueue, args: ['slowFunctional', 'e2e_slow_functional', 1] }), {
      instanceOf: WorkflowFailedError,
    });
  });
});
