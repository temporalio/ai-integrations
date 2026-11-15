/**
 * Activity and Nexus handler runs, matching the Python plugin: no Temporal arguments or results, a status output,
 * cancellation as a status rather than an error, Temporal IDs in Activity metadata, and a propagated parent that
 * keeps tracing on when the env turns it off.
 *
 * @module
 */

import test from 'ava';
import { RunTree } from 'langsmith/run_trees';
import { traceable } from 'langsmith/traceable';
import * as nexus from 'nexus-rpc';
import { CancelledFailure } from '@temporalio/common';

import { createActivityInboundInterceptor, createNexusInboundInterceptor } from '../activity-interceptor';
import { HEADER_KEY, encodeContextString, withContextHeader } from '../propagation';
import { runHeaders } from '../run-tree';
import type { EmitterConfig } from '../sinks';
import { InMemoryRunCollector } from './helpers';

process.env.LANGCHAIN_CALLBACKS_BACKGROUND = 'false';

const ACTIVITY_INFO = {
  activityType: 'act',
  activityId: 'act-1',
  workflowExecution: { workflowId: 'wf-1', runId: 'run-1' },
};

function config(collector: InMemoryRunCollector, addTemporalRuns = true): EmitterConfig {
  return {
    client: collector.asClient(), addTemporalRuns, projectName: 'plugin-project',
    tags: ['plugin-tag'], metadata: { team: 'sdk' },
  };
}

function upstream(): RunTree {
  return new RunTree({ name: 'upstream', project_name: 'upstream-project', client: new InMemoryRunCollector().asClient() });
}

function runActivity(cfg: EmitterConfig, body: () => Promise<unknown>, parent?: RunTree): Promise<unknown> {
  const headers = withContextHeader({}, runHeaders(parent));
  return createActivityInboundInterceptor(cfg)({ info: ACTIVITY_INFO } as never).execute!(
    { args: ['secret-arg'], headers },
    body
  );
}

function startNexus(cfg: EmitterConfig, parent?: RunTree): Promise<unknown> {
  const ctx = runHeaders(parent);
  const headers = ctx ? { [HEADER_KEY]: encodeContextString(ctx) } : {};
  const input = { ctx: { service: 'svc', operation: 'op', headers } } as never;
  return createNexusInboundInterceptor(cfg).startOperation!(input, async () => ({
    result: nexus.HandlerStartOperationResult.sync('secret-result'),
  }));
}

async function withTracingEnv<T>(value: string, fn: () => Promise<T>): Promise<T> {
  const previous = process.env.LANGSMITH_TRACING;
  process.env.LANGSMITH_TRACING = value;
  try {
    return await fn();
  } finally {
    process.env.LANGSMITH_TRACING = previous;
  }
}

test.serial('activity run records no args or result, ends with status ok, and carries Temporal IDs', async (t) => {
  const collector = new InMemoryRunCollector();
  await withTracingEnv('true', () => runActivity(config(collector), async () => 'secret-result'));
  const run = collector.byName('RunActivity:act')!;
  t.deepEqual(run.inputs, {});
  t.deepEqual(run.outputs, { status: 'ok' });
  t.deepEqual(run.tags, ['plugin-tag']);
  t.deepEqual(run.extra?.metadata, {
    team: 'sdk', temporalWorkflowID: 'wf-1', temporalRunID: 'run-1', temporalActivityID: 'act-1',
  });
});

test.serial('nexus run is a tool run that records no input or result and ends with status ok', async (t) => {
  const collector = new InMemoryRunCollector();
  await withTracingEnv('true', () => startNexus(config(collector)));
  const run = collector.byName('RunStartNexusOperationHandler:svc/op')!;
  t.is(run.run_type, 'tool');
  t.deepEqual(run.inputs, {});
  t.deepEqual(run.outputs, { status: 'ok' });
  t.deepEqual(run.extra?.metadata, { team: 'sdk' });
});

test.serial('a cancelled activity ends its run with status cancelled and no error', async (t) => {
  const collector = new InMemoryRunCollector();
  await withTracingEnv('true', () =>
    t.throwsAsync(runActivity(config(collector), async () => { throw new CancelledFailure('cancelled'); }))
  );
  const run = collector.byName('RunActivity:act')!;
  t.deepEqual(run.outputs, { status: 'cancelled' });
  t.is(run.error ?? undefined, undefined);
});

test.serial('a propagated parent keeps activity and nexus tracing on when LANGSMITH_TRACING=false', async (t) => {
  for (const addTemporalRuns of [true, false]) {
    const collector = new InMemoryRunCollector();
    const parent = upstream();
    await withTracingEnv('false', async () => {
      await runActivity(config(collector, addTemporalRuns), traceable(async () => 'x', { name: 'activity-body' }), parent);
      await startNexus(config(collector, addTemporalRuns), parent);
    });
    const names = collector.records.map((run) => run.name);
    t.deepEqual(names, addTemporalRuns
      ? ['RunActivity:act', 'activity-body', 'RunStartNexusOperationHandler:svc/op']
      : ['activity-body']);
    const top = collector.byName(addTemporalRuns ? 'RunActivity:act' : 'activity-body')!;
    t.is(top.parent_run_id, parent.id);
    t.is(top.trace_id, parent.trace_id);
  }
});

test.serial('with no parent, LANGSMITH_TRACING=false emits no activity or nexus runs', async (t) => {
  const collector = new InMemoryRunCollector();
  await withTracingEnv('false', async () => {
    await runActivity(config(collector), traceable(async () => 'x', { name: 'activity-body' }));
    await startNexus(config(collector));
  });
  t.deepEqual(collector.records, []);
});
