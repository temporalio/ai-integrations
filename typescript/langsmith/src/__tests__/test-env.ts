/**
 * Environment tracing gate end-to-end.
 *
 * Tracing is OFF by default and turns on only when a recognized LangSmith
 * tracing env var is explicitly `"true"`. Two suppression paths must emit
 * nothing — no client markers, no workflow-operation runs, no activity runs:
 *  - the explicit tracing gate (`LANGSMITH_TRACING="false"`), and
 *  - the default, with no recognized tracing env var set at all.
 * Each boots a real local environment and runs a workflow with no user
 * `traceable` anywhere, so an empty collector proves the plugin itself emitted
 * nothing rather than merely that there was nothing to emit. A parent propagated in
 * headers keeps tracing on, as in LangSmith.
 *
 * The recognized flags are saved and restored around each case so a
 * non-isolated test pool does not leak state into sibling files or each other.
 * The two cases mutate the same process env, so they run serially.
 *
 * @module
 */

import test from 'ava';
import { RunTree } from 'langsmith/run_trees';
import { Client } from '@temporalio/client';

import { withContextHeader } from '../propagation';
import { runHeaders } from '../run-tree';
import * as activities from './activities/langsmith';
import { InMemoryRunCollector, withTracingWorker, useSharedEnv } from './helpers';
import * as workflows from './workflows/langsmith';

const TRACING_ENV_VARS = ['LANGSMITH_TRACING', 'LANGSMITH_TRACING_V2', 'LANGCHAIN_TRACING_V2', 'LANGCHAIN_TRACING'];

function snapshotTracingEnv(): Record<string, string | undefined> {
  const snapshot: Record<string, string | undefined> = {};
  for (const name of TRACING_ENV_VARS) {
    snapshot[name] = process.env[name];
  }
  return snapshot;
}

function restoreTracingEnv(snapshot: Record<string, string | undefined>): void {
  for (const name of TRACING_ENV_VARS) {
    const value = snapshot[name];
    if (value === undefined) {
      delete process.env[name];
    } else {
      process.env[name] = value;
    }
  }
}

async function runPlainWorkflow(collector: InMemoryRunCollector): Promise<void> {
  await withTracingWorker({
    collector,
    options: { addTemporalRuns: true },
    activities: { plainActivity: activities.plainActivity },
    body: async ({ client, taskQueue }) => {
      await client.workflow.execute(workflows.PlainWorkflow, {
        taskQueue,
        workflowId: `env-${Date.now()}`,
        args: ['hi'],
      });
    },
  });
}

useSharedEnv(test);

test.serial('LANGSMITH_TRACING=false suppresses all emission', async (t) => {
  const snapshot = snapshotTracingEnv();
  process.env.LANGSMITH_TRACING = 'false';
  try {
    const collector = new InMemoryRunCollector();
    await runPlainWorkflow(collector);
    t.deepEqual(collector.records, []);
  } finally {
    restoreTracingEnv(snapshot);
  }
});

test.serial('LANGSMITH_TRACING=false suppresses children built from an untraced run', async (t) => {
  const snapshot = snapshotTracingEnv();
  process.env.LANGSMITH_TRACING = 'false';
  try {
    const collector = new InMemoryRunCollector();
    await withTracingWorker({
      collector,
      options: { projectName: 'project' },
      activities: {},
      body: async ({ client, taskQueue }) => {
        await client.workflow.execute(workflows.ManualChildrenWorkflow, {
          taskQueue,
          workflowId: `env-children-${Date.now()}`,
          workflowExecutionTimeout: '30s',
        });
      },
    });
    t.deepEqual(collector.records, []);
  } finally {
    restoreTracingEnv(snapshot);
  }
});

test.serial('LANGSMITH_TRACING=false still traces under a parent propagated in headers', async (t) => {
  const snapshot = snapshotTracingEnv();
  process.env.LANGSMITH_TRACING = 'false';
  try {
    const collector = new InMemoryRunCollector();
    const parent = new RunTree({ name: 'upstream', project_name: 'upstream-project' });
    await withTracingWorker({
      collector,
      options: { addTemporalRuns: true },
      activities: { plainActivity: activities.plainActivity },
      body: async ({ env, taskQueue }) => {
        // A client without the plugin, so only the propagated header can turn tracing on.
        const client = new Client({
          connection: env.connection,
          namespace: env.namespace,
          interceptors: {
            workflow: [{
              start: (input, next) => next({ ...input, headers: withContextHeader(input.headers, runHeaders(parent)) }),
            }],
          },
        });
        await client.workflow.execute(workflows.PlainWorkflow, {
          taskQueue,
          workflowId: `env-parent-${Date.now()}`,
          args: ['hi'],
          workflowExecutionTimeout: '30s',
        });
      },
    });
    t.deepEqual(collector.records.map((run) => run.name).sort(), [
      'RunActivity:plainActivity', 'RunWorkflow:PlainWorkflow', 'StartActivity:plainActivity',
    ]);
    t.is(collector.byName('RunWorkflow:PlainWorkflow')?.parent_run_id, parent.id);
    t.is(collector.parentNameOf('RunActivity:plainActivity'), 'RunWorkflow:PlainWorkflow');
    t.true(collector.records.every((run) => run.trace_id === parent.trace_id && run.end_time !== undefined));
  } finally {
    restoreTracingEnv(snapshot);
  }
});

test.serial('default-off: suppresses all emission when no recognized tracing env var is set', async (t) => {
  const snapshot = snapshotTracingEnv();
  for (const name of TRACING_ENV_VARS) {
    delete process.env[name];
  }
  try {
    const collector = new InMemoryRunCollector();
    await runPlainWorkflow(collector);
    t.deepEqual(collector.records, []);
  } finally {
    restoreTracingEnv(snapshot);
  }
});
