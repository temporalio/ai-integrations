import test from 'ava';
import { LangGraphPlugin } from '../index';
import { registry } from './graphs';
import { activitySummaries, execute, replay, setupTestEnv, uid, withWorker } from './helpers';
import type { HistoryEvent } from './helpers';
import { runGraph } from './workflows';

const env = setupTestEnv(test);

async function run(key: string, input: string, workflowId = uid('summary')): Promise<unknown> {
  const taskQueue = uid('summary');
  return withWorker(env(), { taskQueue, plugin: new LangGraphPlugin(registry[key]!), maxCachedWorkflows: 0 }, () =>
    execute(env(), runGraph, { taskQueue, workflowId, args: [key, 'summary-graph', { value: input }] })
  );
}

async function summaries(key: string, input: string): Promise<string[]> {
  const workflowId = uid('summary');
  await run(key, input, workflowId);
  const { events } = await env().client.workflow.getHandle(workflowId).fetchHistory();
  return activitySummaries((events ?? []) as HistoryEvent[]);
}

for (const [name, key, input, expected] of [
  ['summaryFn sets the Activity summary', 'summaries.summaryFn', 'hello', '"value=hello"'],
  ['a null summaryFn result sets none', 'summaries.summaryFnNull', 'x', ''],
  ['an empty summaryFn result sets none', 'summaries.summaryFnEmpty', 'x', ''],
  ['a static summary', 'summaries.staticSummary', 'x', '"static"'],
  ["a node's summary overrides the default summaryFn", 'summaries.nodeStaticOverDefaultFn', 'x', '"node-static"'],
  ["a node's summaryFn overrides the default summary", 'summaries.nodeFnOverDefaultStatic', 'x', '"node-fn"'],
  ['the default summaryFn applies without an override', 'summaries.defaultFn', 'x', '"global"'],
] as const) {
  test(name, async (t) => {
    t.deepEqual(await summaries(key, input), [expected]);
  });
}

test('summaryFn is not in the node metadata', async (t) => {
  t.deepEqual(await run('summaries.hidesSummaryFn', 'in'), { value: 'in-has_fn=false' });
});

/** Runs a Workflow-node graph, then queries the completed Workflow's current details. */
async function currentDetails(key: string): Promise<{ result: unknown; details: string | undefined }> {
  const taskQueue = uid('wf-node');
  const workflowId = uid('wf-node');
  return withWorker(env(), { taskQueue, plugin: new LangGraphPlugin(registry[key]!) }, async () => {
    const result = await execute(env(), runGraph, {
      taskQueue,
      workflowId,
      args: [key, 'summary-graph', { value: 'ready' }],
    });
    // Workflow nodes set the details last-writer-wins and nothing clears them at
    // the end of the graph, so the completed Workflow answers like a mid-run query.
    const metadata = await env()
      .client.workflow.getHandle(workflowId)
      .query<{ currentDetails?: string }>('__temporal_workflow_metadata');
    return { result, details: metadata.currentDetails };
  });
}

test('a Workflow node sets the current details', async (t) => {
  t.deepEqual(await currentDetails('summaries.workflowNode'), { result: { value: 'ready' }, details: 'wf:ready' });
});

test('a Workflow node clears the current details with an empty summary', async (t) => {
  const { result, details } = await currentDetails('summaries.workflowNodeClears');
  t.deepEqual(result, { value: 'ready' });
  t.falsy(details);
});

test('a run with summaryFn replays', async (t) => {
  const workflowId = uid('summary-replay');
  await run('summaries.summaryFn', 'hello', workflowId);
  await replay(env(), new LangGraphPlugin(registry['summaries.summaryFn']!), workflowId);
  t.pass();
});
