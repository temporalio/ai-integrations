/**
 * Workflow-side signal markers.
 *
 * When a workflow signals another workflow from inside its body, the outbound
 * `signalWorkflow` interceptor emits a closed `SignalChildWorkflow:` or
 * `SignalExternalWorkflow:` marker and propagates the caller's run, so the
 * signalled workflow's `HandleSignal:` run is the marker's sibling.
 *
 * @module
 */

import test from 'ava';

import * as activities from './activities/langsmith';
import { InMemoryRunCollector, withTracingWorker, useSharedEnv } from './helpers';
import * as workflows from './workflows/langsmith';

process.env.LANGSMITH_TRACING = 'true';

useSharedEnv(test);

for (const [workflow, marker] of [
  [workflows.SignalChildWorkflow, 'SignalChildWorkflow:my_signal'],
  [workflows.SignalExternalWorkflow, 'SignalExternalWorkflow:my_signal'],
] as const) {
  test(`${workflow.name}: emits a closed ${marker} marker with the handler as its sibling`, async (t) => {
    const collector = new InMemoryRunCollector();
    const result = await withTracingWorker({
      collector,
      options: { addTemporalRuns: true, tags: ['plugin-tag'], metadata: { team: 'sdk' } },
      activities: { simpleActivity: activities.simpleActivity },
      workerOptions: { maxCachedWorkflows: 2 },
      body: async ({ client, taskQueue }) =>
        client.workflow.execute(workflow, {
          taskQueue,
          workflowId: `${workflow.name}-${Date.now()}`,
          args: ['hello'],
        }),
    });

    t.is(result, 'hello');

    const run = collector.byName(marker);
    t.truthy(run, `a ${marker} marker run was emitted`);
    t.not(run!.end_time, undefined, 'marker run has an end_time (was closed)');
    t.deepEqual(run!.inputs, {});
    t.deepEqual(run!.outputs, { status: 'ok' });
    t.deepEqual(run!.tags, ['plugin-tag']);
    t.is((run!.extra?.metadata as Record<string, unknown>).team, 'sdk');
    t.is(collector.parentNameOf(marker), `RunWorkflow:${workflow.name}`);
    t.is(collector.parentNameOf('HandleSignal:my_signal'), `RunWorkflow:${workflow.name}`);
  });
}
