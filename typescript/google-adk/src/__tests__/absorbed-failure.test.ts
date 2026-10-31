/**
 * Terminal-outcome tests for model failures that `@google/adk` absorbs. Because the
 * absorbed failure lets the run finish normally, every case here asserts the
 * *execution's* status, not just what the Workflow returned: a failed model call must
 * FAIL the execution, a cancel must end it CANCELLED, and a failure the caller
 * handled must leave it COMPLETED. ADK absorbs a cancel in a tool or a model callback
 * the same way, and that one must end the execution CANCELLED too, after the main
 * function's cleanup when a Signal handler ran the turn. Only the execution's own cancel
 * counts: neither one a timeout scope caused nor an ordinary error may decide a handled
 * cancellation, wherever the Workflow shields its waits.
 */

import test from 'ava';

import { ApplicationFailure, TimeoutFailure } from '@temporalio/common';

import { GoogleAdkPlugin } from '../index';
import {
  countScheduledActivities,
  defaultTestProvider,
  findInCauseChain,
  setupTestEnv,
  uid,
  waitForScheduledActivities,
  withWorker,
} from './helpers';
import { cancellableActivity, echoId } from './test-activities';
import {
  adkAwaitSignal,
  adkChatUpdate,
  adkContinueUpdate,
  adkDoneSignal,
  adkRecoverSignal,
  adkStartSignal,
  adkStageQuery,
  agentRunnerAwaitedSignalTurn,
  agentRunnerBlockedTurn,
  agentRunnerCaughtCancelThenFailingTurn,
  agentRunnerCaughtCancelThenTimedOutTurn,
  agentRunnerContinueAsNew,
  agentRunnerContinueAsNewFromUpdate,
  agentRunnerFailThenSlowModel,
  agentRunnerFailureAfterTimeoutScope,
  agentRunnerFailureThenRecoveringSignal,
  agentRunnerFailureWithCompensation,
  agentRunnerOneTurn,
  agentRunnerRecoversFromCancelledModel,
  agentRunnerRecoversFromModelError,
  agentRunnerRecoversOnlyTheSecondFailure,
  agentRunnerShieldedWaitThenCaughtCancel,
  agentRunnerSignalTurnThenCleanup,
  agentRunnerThrowingSummary,
  agentRunnerTimedOutTurnThenCaughtCancel,
  agentRunnerTurnUnderTimeoutScope,
  agentRunnerUnawaitedSignalTurn,
  agentRunnerUpdateDriven,
  caughtModelCallError,
} from './workflows';

const getEnv = setupTestEnv(test);

function adkPlugin(): GoogleAdkPlugin {
  return new GoogleAdkPlugin({ modelProvider: defaultTestProvider() });
}

test.serial('modelFailureThroughTheRunnerFailsTheWorkflow', async (t) => {
  const env = getEnv();
  const taskQueue = uid('adk-agent-fail');
  const workflowId = uid('wf-agent-fail');
  await withWorker(env, { taskQueue, plugins: [adkPlugin()] }, async () => {
    const handle = await env.client.workflow.start(agentRunnerOneTurn, {
      taskQueue,
      workflowId,
      args: ['boom', 'explode'],
    });
    const caught = await t.throwsAsync(handle.result());
    const appFailure = findInCauseChain(caught, ApplicationFailure);
    t.is(appFailure?.type, 'GoogleAdkModelError.400');
    t.is((await handle.describe()).status.name, 'FAILED');
  });
});

test.serial('cancellingARunnerDrivenAgentEndsCancelled', async (t) => {
  const env = getEnv();
  const taskQueue = uid('adk-agent-cancel');
  const workflowId = uid('wf-agent-cancel');
  await withWorker(env, { taskQueue, plugins: [adkPlugin()] }, async () => {
    const handle = await env.client.workflow.start(agentRunnerOneTurn, {
      taskQueue,
      workflowId,
      args: ['slow-model', 'wait'],
    });
    // Cancel only once the model Activity is scheduled, so it rejects with an
    // `ActivityFailure` carrying a `CancelledFailure` cause.
    await waitForScheduledActivities(env, workflowId, 'adk-invokeModel');
    await handle.cancel();
    await t.throwsAsync(handle.result());
    t.is((await handle.describe()).status.name, 'CANCELLED');
  });
});

test.serial('cancellingAfterAnAbsorbedFailureStillEndsCancelled', async (t) => {
  const env = getEnv();
  const taskQueue = uid('adk-agent-fail-cancel');
  const workflowId = uid('wf-agent-fail-cancel');
  await withWorker(env, { taskQueue, plugins: [adkPlugin()] }, async () => {
    const handle = await env.client.workflow.start(agentRunnerFailThenSlowModel, {
      taskQueue,
      workflowId,
      args: ['go'],
    });
    // Two scheduled model Activities means the failing turn is behind us, so the
    // cancellation arrives with a failure already absorbed.
    await waitForScheduledActivities(env, workflowId, 'adk-invokeModel', 2);
    await handle.cancel();
    await t.throwsAsync(handle.result());
    t.is((await handle.describe()).status.name, 'CANCELLED');
  });
});

test.serial('onModelErrorRecoveryLeavesTheWorkflowCompleted', async (t) => {
  const env = getEnv();
  const taskQueue = uid('adk-agent-recover');
  const workflowId = uid('wf-agent-recover');
  await withWorker(env, { taskQueue, plugins: [adkPlugin()] }, async () => {
    const result = await env.client.workflow.execute(agentRunnerRecoversFromModelError, {
      taskQueue,
      workflowId,
      args: ['explode'],
    });
    t.is(result, 'recovered');
    t.is((await env.client.workflow.getHandle(workflowId).describe()).status.name, 'COMPLETED');
  });
});

test.serial('onModelErrorRecoveryLeavesAnEarlierFailureStanding', async (t) => {
  const env = getEnv();
  const taskQueue = uid('adk-agent-recover-second');
  const workflowId = uid('wf-agent-recover-second');
  await withWorker(env, { taskQueue, plugins: [adkPlugin()] }, async () => {
    const handle = await env.client.workflow.start(agentRunnerRecoversOnlyTheSecondFailure, {
      taskQueue,
      workflowId,
      args: ['explode'],
    });
    const caught = await t.throwsAsync(handle.result());
    t.is(findInCauseChain(caught, ApplicationFailure)?.type, 'GoogleAdkModelError.400');
    t.is((await handle.describe()).status.name, 'FAILED');
  });
});

test.serial('cancellingARunnerWhoseCallbackRecoversStillEndsCancelled', async (t) => {
  const env = getEnv();
  const taskQueue = uid('adk-agent-cancel-recover');
  const workflowId = uid('wf-agent-cancel-recover');
  await withWorker(env, { taskQueue, plugins: [adkPlugin()] }, async () => {
    const handle = await env.client.workflow.start(agentRunnerRecoversFromCancelledModel, {
      taskQueue,
      workflowId,
      args: ['wait'],
    });
    // The cancel reaches `onModelErrorCallback` as any other model failure does, and that
    // callback recovers from it — but a cancelled execution is not the caller's to absorb.
    await waitForScheduledActivities(env, workflowId, 'adk-invokeModel');
    await handle.cancel();
    await t.throwsAsync(handle.result());
    t.is((await handle.describe()).status.name, 'CANCELLED');
  });
});

/** Resolves once the Workflow's {@link adkStageQuery} answers `stage`. */
async function waitForStage(
  handle: { query(query: typeof adkStageQuery): Promise<string> },
  stage: string
): Promise<void> {
  const deadline = Date.now() + 10_000;
  while ((await handle.query(adkStageQuery)) !== stage) {
    if (Date.now() > deadline) throw new Error(`timed out waiting for stage ${stage}`);
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
}

/** Resolves once history shows a `NexusOperationScheduled` event. */
async function waitForScheduledNexusOperation(handle: {
  fetchHistory(): Promise<{ events?: Array<{ nexusOperationScheduledEventAttributes?: unknown }> | null }>;
}): Promise<void> {
  const deadline = Date.now() + 10_000;
  while (!((await handle.fetchHistory()).events ?? []).some((e) => e.nexusOperationScheduledEventAttributes)) {
    if (Date.now() > deadline) throw new Error('timed out waiting for a scheduled Nexus operation');
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
}

test.serial('cancellingAToolOrModelCallbackWaitEndsCancelled', async (t) => {
  const env = getEnv();
  // ADK absorbs these cancels without a model call that fails after them: it drops a
  // long-running tool's error before 2.3, and from 2.3 the tool's `skipSummarization` ends
  // the turn on the error's function response. It turns a callback's cancelled Activity
  // into an error event, a plugin's first wrapped in a plain `Error`, that ends the turn.
  // The same waits on a Nexus operation that never starts fail with the operation's start,
  // not its result.
  const endpoint = uid('adk-unpolled-endpoint');
  const nexusEndpoint = await env.createNexusEndpoint(endpoint, uid('adk-unpolled'));
  t.teardown(() => env.deleteNexusEndpoint(nexusEndpoint));
  for (const nexus of [false, true]) {
    for (const kind of ['tool', 'agent', 'plugin'] as const) {
      const label = nexus ? `${kind} nexus` : kind;
      const taskQueue = uid(`adk-${kind}-wait-cancel`);
      const workflowId = uid(`wf-${kind}-wait-cancel`);
      await withWorker(env, { taskQueue, plugins: [adkPlugin()], activities: { cancellableActivity } }, async () => {
        const handle = await env.client.workflow.start(agentRunnerBlockedTurn, {
          taskQueue,
          workflowId,
          args: nexus ? [kind, endpoint] : [kind],
        });
        await waitForStage(handle, 'blocked');
        if (nexus) await waitForScheduledNexusOperation(handle);
        else if (kind !== 'tool') await waitForScheduledActivities(env, workflowId, 'cancellableActivity');
        await handle.cancel();
        await t.throwsAsync(handle.result());
        t.is((await handle.describe()).status.name, 'CANCELLED', label);
      });
    }
  }
});

test.serial('cancellingASignalDrivenTurnLetsTheMainFunctionCleanUpFirst', async (t) => {
  const env = getEnv();
  // With `failFirst`, the handler absorbed a model failure before the cancel; the cancel
  // still outranks it and must not leave it behind to end the execution FAILED.
  for (const failFirst of [false, true]) {
    const taskQueue = uid('adk-signal-turn-cleanup');
    const workflowId = uid('wf-signal-turn-cleanup');
    await withWorker(env, { taskQueue, plugins: [adkPlugin()], activities: { echoId } }, async () => {
      const handle = await env.client.workflow.start(agentRunnerSignalTurnThenCleanup, {
        taskQueue,
        workflowId,
        args: [failFirst],
      });
      await handle.signal(adkStartSignal);
      await waitForStage(handle, 'blocked');
      await handle.cancel();
      // The Signal handler's absorbed cancel waits for the main function, which is still
      // cleaning up between its two Activities.
      await waitForStage(handle, 'held');
      t.is((await handle.describe()).status.name, 'RUNNING', `failFirst=${failFirst}`);
      await handle.signal(adkDoneSignal);
      await t.throwsAsync(handle.result());
      t.is((await handle.describe()).status.name, 'CANCELLED', `failFirst=${failFirst}`);
      const { events } = await handle.fetchHistory();
      t.is(countScheduledActivities(events ?? [], 'echoId'), 2, `failFirst=${failFirst}`);
    });
  }
});

test.serial('aTimedOutTurnLeavesALaterCaughtCancelCompleted', async (t) => {
  const env = getEnv();
  // The turn absorbs only its timeout scope's cancel, or an ordinary error. The Workflow's own
  // cancellation never reaches what the turn absorbed, and the Workflow handles it: after the
  // turn, before a turn shielded from it, while a tool or callback shields its own wait or
  // the cleanup after it, also a wait on a Nexus operation that never starts, or before a
  // turn whose callback fails on its own.
  const endpoint = uid('adk-unpolled-endpoint');
  const nexusEndpoint = await env.createNexusEndpoint(endpoint, uid('adk-unpolled'));
  t.teardown(() => env.deleteNexusEndpoint(nexusEndpoint));
  const all = ['tool', 'agent', 'plugin', 'model'] as const;
  const operations = ['tool', 'agent', 'plugin'] as const;
  const cases = [
    ...all.map((kind) => ({ workflow: agentRunnerTimedOutTurnThenCaughtCancel, args: [kind], cancelAt: 'after-turn' })),
    ...all.map((kind) => ({ workflow: agentRunnerCaughtCancelThenTimedOutTurn, args: [kind], cancelAt: 'waiting' })),
    ...operations.flatMap((kind) => [
      { workflow: agentRunnerShieldedWaitThenCaughtCancel, args: [kind, 'shielded'], cancelAt: 'blocked' },
      { workflow: agentRunnerShieldedWaitThenCaughtCancel, args: [kind, 'shielded-cleanup'], cancelAt: 'cleanup' },
      { workflow: agentRunnerShieldedWaitThenCaughtCancel, args: [kind, 'shielded', endpoint], cancelAt: 'blocked' },
    ]),
    ...(['agent', 'plugin'] as const).map((kind) => ({
      workflow: agentRunnerCaughtCancelThenFailingTurn,
      args: [kind],
      cancelAt: 'waiting',
    })),
  ];
  for (const { workflow, args, cancelAt } of cases) {
    const label = `${workflow.name}(${args.join(', ')})`;
    const taskQueue = uid(`adk-${args[0]}-timeout-and-cancel`);
    const workflowId = uid(`wf-${args[0]}-timeout-and-cancel`);
    await withWorker(env, { taskQueue, plugins: [adkPlugin()], activities: { cancellableActivity } }, async () => {
      const handle = await env.client.workflow.start(workflow as (...args: string[]) => Promise<string>, {
        taskQueue,
        workflowId,
        args,
      });
      await waitForStage(handle, cancelAt);
      await handle.cancel();
      // A cleanup that holds the timeout's error lets it reach ADK only after the cancel.
      if (cancelAt === 'cleanup') await handle.signal(adkDoneSignal);
      t.is(await handle.result(), 'caught', label);
      t.is((await handle.describe()).status.name, 'COMPLETED', label);
    });
  }
});

test.serial('throwingSummaryCallbackFailsTheWorkflow', async (t) => {
  const env = getEnv();
  const taskQueue = uid('adk-agent-summary');
  const workflowId = uid('wf-agent-summary');
  await withWorker(env, { taskQueue, plugins: [adkPlugin()] }, async () => {
    const handle = await env.client.workflow.start(agentRunnerThrowingSummary, {
      taskQueue,
      workflowId,
      args: ['hi'],
    });
    const caught = await t.throwsAsync(handle.result());
    t.is(findInCauseChain(caught, ApplicationFailure)?.type, 'TestSummaryFailure');
    t.is((await handle.describe()).status.name, 'FAILED');
  });
});

test.serial('handledDirectModelFailureLeavesTheWorkflowCompleted', async (t) => {
  const env = getEnv();
  const taskQueue = uid('adk-direct-caught');
  const workflowId = uid('wf-direct-caught');
  await withWorker(env, { taskQueue, plugins: [adkPlugin()] }, async () => {
    const result = await env.client.workflow.execute(caughtModelCallError, { taskQueue, workflowId });
    t.is(result, 'GoogleAdkModelError.400');
    t.is((await env.client.workflow.getHandle(workflowId).describe()).status.name, 'COMPLETED');
  });
});

test.serial('aTimeoutScopeAroundAnAgentTurnLeavesTheWorkflowCompleted', async (t) => {
  const env = getEnv();
  const taskQueue = uid('adk-agent-timeout-uncaught');
  const workflowId = uid('wf-agent-timeout-uncaught');
  await withWorker(env, { taskQueue, plugins: [adkPlugin()] }, async () => {
    const result = await env.client.workflow.execute(agentRunnerTurnUnderTimeoutScope, {
      taskQueue,
      workflowId,
      args: ['wait'],
    });
    t.is(result, 'timed out');
    t.is((await env.client.workflow.getHandle(workflowId).describe()).status.name, 'COMPLETED');
  });
});

test.serial('modelFailureAfterATimeoutScopedTurnFailsTheWorkflow', async (t) => {
  const env = getEnv();
  const taskQueue = uid('adk-agent-fail-after-scope');
  const workflowId = uid('wf-agent-fail-after-scope');
  await withWorker(env, { taskQueue, plugins: [adkPlugin()] }, async () => {
    const handle = await env.client.workflow.start(agentRunnerFailureAfterTimeoutScope, {
      taskQueue,
      workflowId,
      args: ['wait'],
    });
    const caught = await t.throwsAsync(handle.result());
    t.is(findInCauseChain(caught, ApplicationFailure)?.type, 'GoogleAdkModelError.400');
    t.is((await handle.describe()).status.name, 'FAILED');
  });
});

test.serial('modelFailureInAnUpdateRejectsThatUpdateOnly', async (t) => {
  const env = getEnv();
  const taskQueue = uid('adk-agent-update');
  const workflowId = uid('wf-agent-update');
  await withWorker(env, { taskQueue, plugins: [adkPlugin()] }, async () => {
    const handle = await env.client.workflow.start(agentRunnerUpdateDriven, { taskQueue, workflowId });

    const caught = await t.throwsAsync(handle.executeUpdate(adkChatUpdate, { args: ['boom'] }));
    t.is(findInCauseChain(caught, ApplicationFailure)?.type, 'GoogleAdkModelError.400');

    // The rejected Update's frame consumed its own failure: it leaked into neither the
    // next Update's frame nor the main function's frame.
    t.is(await handle.executeUpdate(adkChatUpdate, { args: ['fake-model'] }), 'fake-response:fake-model');
    await handle.signal(adkDoneSignal);
    await handle.result();
    t.is((await handle.describe()).status.name, 'COMPLETED');
  });
});

test.serial('modelFailureBeforeContinueAsNewFailsTheWorkflow', async (t) => {
  const env = getEnv();
  const taskQueue = uid('adk-agent-can');
  const workflowId = uid('wf-agent-can');
  await withWorker(env, { taskQueue, plugins: [adkPlugin()] }, async () => {
    // The next run would use a model that answers, so a Workflow that continues as new
    // here would otherwise reach a green terminal state despite the failed turn.
    const handle = await env.client.workflow.start(agentRunnerContinueAsNew, {
      taskQueue,
      workflowId,
      args: ['boom', 'fake-model'],
    });
    const caught = await t.throwsAsync(handle.result());
    t.is(findInCauseChain(caught, ApplicationFailure)?.type, 'GoogleAdkModelError.400');
    t.is((await handle.describe()).status.name, 'FAILED');
  });
});

test.serial('modelFailureRecoveredInASignalLeavesTheMainFunctionsStanding', async (t) => {
  const env = getEnv();
  const taskQueue = uid('adk-agent-signal-recover');
  const workflowId = uid('wf-agent-signal-recover');
  await withWorker(env, { taskQueue, plugins: [adkPlugin()] }, async () => {
    const handle = await env.client.workflow.start(agentRunnerFailureThenRecoveringSignal, {
      taskQueue,
      workflowId,
    });
    // Two scheduled model Activities means the main function's failing turn is behind us,
    // so the Signal's own turn is what the recovery in it applies to.
    await waitForScheduledActivities(env, workflowId, 'adk-invokeModel', 2);
    await handle.signal(adkRecoverSignal);
    const caught = await t.throwsAsync(handle.result());
    // A timeout is the main function's own failure; the turn its Signal recovered was a 400.
    t.truthy(findInCauseChain(caught, TimeoutFailure));
    t.is((await handle.describe()).status.name, 'FAILED');
  });
});

test.serial('modelFailureInAnAwaitedSignalTurnFailsTheWorkflow', async (t) => {
  const env = getEnv();
  const taskQueue = uid('adk-agent-signal-awaited');
  const workflowId = uid('wf-agent-signal-awaited');
  await withWorker(env, { taskQueue, plugins: [adkPlugin()] }, async () => {
    const handle = await env.client.workflow.start(agentRunnerAwaitedSignalTurn, { taskQueue, workflowId });
    await handle.signal(adkAwaitSignal);
    const caught = await t.throwsAsync(handle.result());
    t.is(findInCauseChain(caught, ApplicationFailure)?.type, 'GoogleAdkModelError.400');
    t.is((await handle.describe()).status.name, 'FAILED');
  });
});

test.serial('modelFailureFromAnUnawaitedSignalTurnFailsTheWorkflow', async (t) => {
  const env = getEnv();
  const taskQueue = uid('adk-agent-signal-unawaited');
  const workflowId = uid('wf-agent-signal-unawaited');
  await withWorker(env, { taskQueue, plugins: [adkPlugin()] }, async () => {
    const handle = await env.client.workflow.start(agentRunnerUnawaitedSignalTurn, { taskQueue, workflowId });
    await handle.signal(adkStartSignal);
    const caught = await t.throwsAsync(handle.result());
    t.is(findInCauseChain(caught, ApplicationFailure)?.type, 'GoogleAdkModelError.400');
    t.is((await handle.describe()).status.name, 'FAILED');
  });
});

test.serial('anUnrelatedSignalLeavesTheMainFunctionToCompensate', async (t) => {
  const env = getEnv();
  const taskQueue = uid('adk-agent-signal-compensate');
  const workflowId = uid('wf-agent-signal-compensate');
  const activities = { compensate: async () => undefined };
  await withWorker(env, { taskQueue, plugins: [adkPlugin()], activities }, async () => {
    const handle = await env.client.workflow.start(agentRunnerFailureWithCompensation, { taskQueue, workflowId });
    await waitForScheduledActivities(env, workflowId, 'adk-invokeModel', 2);
    await handle.signal(adkDoneSignal);
    const caught = await t.throwsAsync(handle.result());
    t.is(findInCauseChain(caught, ApplicationFailure)?.type, 'GoogleAdkModelError.400');
    t.is((await handle.describe()).status.name, 'FAILED');
    // The Signal absorbed nothing, so the failure stayed in the main function's frame and
    // that function reached its `finally` before the execution failed.
    const { events } = await handle.fetchHistory();
    t.is(countScheduledActivities(events ?? [], 'compensate'), 1);
  });
});

test.serial('continueAsNewFromAnUpdateFailsOnTheMainFunctionsFailure', async (t) => {
  const env = getEnv();
  const taskQueue = uid('adk-agent-can-update');
  const workflowId = uid('wf-agent-can-update');
  await withWorker(env, { taskQueue, plugins: [adkPlugin()] }, async () => {
    const handle = await env.client.workflow.start(agentRunnerContinueAsNewFromUpdate, {
      taskQueue,
      workflowId,
      args: ['boom'],
    });
    await waitForScheduledActivities(env, workflowId, 'adk-invokeModel', 2);
    const rejected = await t.throwsAsync(handle.executeUpdate(adkContinueUpdate));
    t.is(findInCauseChain(rejected, ApplicationFailure)?.type, 'GoogleAdkModelError.400');

    await handle.signal(adkDoneSignal);
    const caught = await t.throwsAsync(handle.result());
    t.is(findInCauseChain(caught, ApplicationFailure)?.type, 'GoogleAdkModelError.400');
    t.is((await handle.describe()).status.name, 'FAILED');
  });
});
