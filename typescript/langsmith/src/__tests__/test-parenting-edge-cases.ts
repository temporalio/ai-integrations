/**
 * Run-hierarchy edge cases the comprehensive test cannot cover: no-ambient-run roots,
 * the root workflow-body `traceable` `crypto` regression, and per-field plugin options.
 *
 * @module
 */

import test from 'ava';

import * as activities from './activities/langsmith';
import { InMemoryRunCollector, SIMPLE_TREE, dumpTraces, withTracingWorker, useSharedEnv } from './helpers';
import * as workflows from './workflows/langsmith';

process.env.LANGSMITH_TRACING = 'true';

const ALL_ACTIVITIES = {
  simpleActivity: activities.simpleActivity,
};

/** Root workflow-body `traceable`, no propagated parent — just the user run, no placeholder root. */
const WORKFLOW_BODY_ROOT_TREE = ['workflow_inner_call'].join('\n');

useSharedEnv(test);

test('emits the basic SimpleWorkflow tree with no ambient (two roots)', async (t) => {
  const collector = new InMemoryRunCollector();
  await withTracingWorker({
    collector,
    options: { addTemporalRuns: true },
    activities: ALL_ACTIVITIES,
    body: async ({ client, taskQueue }) => {
      await client.workflow.execute(workflows.SimpleWorkflow, {
        taskQueue,
        workflowId: `simple-${Date.now()}`,
        args: ['hi'],
      });
    },
  });
  t.deepEqual(dumpTraces(collector.records), SIMPLE_TREE);
});

// Native root construction uses deterministic defaults without calling crypto inside the isolate.
test('root workflow-body traceable (no propagated parent) does not crash and emits just the user run', async (t) => {
  const collector = new InMemoryRunCollector();
  await withTracingWorker({
    collector,
    options: { addTemporalRuns: false },
    activities: ALL_ACTIVITIES,
    body: async ({ client, taskQueue }) => {
      await client.workflow.execute(workflows.WorkflowBodyTraceableWorkflow, {
        taskQueue,
        workflowId: `wf-body-root-${Date.now()}`,
        args: ['hello'],
      });
    },
  });
  t.deepEqual(dumpTraces(collector.records), WORKFLOW_BODY_ROOT_TREE);
});

test('plugin options and Temporal IDs are carried onto plugin runs, which record no args or results', async (t) => {
  const collector = new InMemoryRunCollector();
  const workflowId = `options-${Date.now()}`;
  const runId = await withTracingWorker({
    collector,
    options: {
      addTemporalRuns: true,
      projectName: 'my-project',
      tags: ['env:test'],
      // Sent unchanged, as the Python plugin does.
      metadata: { team: 'platform', api_key: 'kept' },
    },
    activities: ALL_ACTIVITIES,
    body: async ({ client, taskQueue }) => {
      const handle = await client.workflow.start(workflows.SimpleWorkflow, { taskQueue, workflowId, args: ['hi'] });
      await handle.result();
      return handle.firstExecutionRunId;
    },
  });

  const runWorkflow = collector.byName('RunWorkflow:SimpleWorkflow');
  t.is(runWorkflow?.project_name, 'my-project');
  t.deepEqual(runWorkflow?.tags, ['env:test']);
  t.deepEqual(runWorkflow?.extra?.metadata, {
    team: 'platform', api_key: 'kept', temporalWorkflowID: workflowId, temporalRunID: runId,
  });

  const runActivity = collector.byName('RunActivity:simpleActivity');
  t.deepEqual(runActivity?.extra?.metadata, {
    team: 'platform', api_key: 'kept', temporalWorkflowID: workflowId, temporalRunID: runId, temporalActivityID: '1',
  });

  const marker = collector.byName('StartActivity:simpleActivity');
  t.deepEqual(marker?.tags, ['env:test']);
  t.like(marker?.extra?.metadata, { team: 'platform', api_key: 'kept' });

  for (const run of [runWorkflow, runActivity, marker]) {
    t.deepEqual(run?.inputs, {}, run?.name);
    t.deepEqual(run?.outputs, { status: 'ok' }, run?.name);
  }
});
