import { randomUUID } from 'node:crypto';
import test from 'ava';
import { address } from 'langsmith';
import { RunTree, type WriteReplica } from 'langsmith/run_trees';
import { traceable } from 'langsmith/traceable';
import { Worker } from '@temporalio/worker';
import { LangSmithPlugin } from '../index';
import { runHeaders, runTreeFromContext } from '../run-tree';
import { bindReplicaCredentials } from '../routing';
import { HEADER_KEY } from '../propagation';
import * as activities from './activities/routing';
import { InMemoryRunCollector, useSharedEnv, withTracingWorker } from './helpers';
import { comprehensiveNexusServiceHandler } from './stubs/nexus';
import { ROUTING_ADDRESS, ROUTING_ENDPOINT, RoutingWorkflow } from './workflows/routing';

process.env.LANGSMITH_TRACING = 'true';
process.env.LANGCHAIN_CALLBACKS_BACKGROUND = 'false';
const WORKFLOWS_PATH = require.resolve('./workflows/routing');
const SECONDARY_ADDRESS = address.agent('support', 'production');
const REPLICAS: WriteReplica[] = [
  { projectName: 'primary', primary: true },
  { projectName: 'secondary', updates: { tags: ['secondary-only'] } },
  { address: SECONDARY_ADDRESS, updates: { extra: { metadata: { deployment: 'production' } } } },
];

useSharedEnv(test);

test('plugin validates mutually exclusive destinations', (t) => {
  t.throws(() => new LangSmithPlugin({ projectName: 'project', address: ROUTING_ADDRESS }), {
    message: 'projectName and address are mutually exclusive',
  });
  t.throws(() => new LangSmithPlugin({ address: 'invalid' as typeof ROUTING_ADDRESS }));
});

test('headers preserve addresses, replica routing and parent timestamps without credentials', (t) => {
  const collector = new InMemoryRunCollector();
  const time = Date.parse('2024-01-02T03:04:05.678Z');
  const parent = new RunTree({
    name: 'parent',
    id: randomUUID(),
    start_time: time,
    address: ROUTING_ADDRESS,
    client: collector.asClient(),
    extra: { metadata: { team: 'sdk', api_key: 'metadata-secret' } },
    replicas: [{
      address: SECONDARY_ADDRESS,
      primary: true,
      apiKey: 'replica-secret',
      apiUrl: 'https://private-ingestion.invalid',
      workspaceId: 'private-workspace',
      client: collector.asClient(),
      updates: {
        tags: ['replicated'], extra: { metadata: { team: 'sdk', token: 'nested-secret' } },
        id: 'untrusted-id', address: 'untrusted-destination',
      },
    }],
  });
  const headers = runHeaders(parent)!;
  const encoded = decodeURIComponent(headers.baggage);
  for (const secret of ['metadata-secret', 'replica-secret', 'private-ingestion', 'private-workspace', 'nested-secret']) {
    t.false(encoded.includes(secret));
  }
  t.false(encoded.includes('untrusted-id'));
  t.false(encoded.includes('untrusted-destination'));
  const restored = runTreeFromContext(headers)!;
  t.is(restored.address, ROUTING_ADDRESS);
  t.is(restored.project_name, undefined);
  t.is(restored.start_time, time);
  t.deepEqual(restored.replicas, [{
    address: SECONDARY_ADDRESS,
    primary: true,
    updates: { tags: ['replicated'], extra: { metadata: { team: 'sdk' } } },
  }]);
});

test('replica headers support Python and JavaScript project names and explicit empty routing', (t) => {
  const parent = new RunTree({ name: 'parent', replicas: [{ projectName: 'audit', primary: true }] });
  const headers = runHeaders(parent)!;
  t.true(decodeURIComponent(headers.baggage).includes('"project_name":"audit"'));
  t.is(runTreeFromContext(headers)?.replicas?.[0]?.projectName, 'audit');
  const legacy = { ...headers, baggage: headers.baggage.replace('project_name', 'projectName') };
  t.is(runTreeFromContext(legacy)?.replicas?.[0]?.projectName, 'audit');
  const tuple = { ...headers, baggage: `langsmith-replicas=${encodeURIComponent(JSON.stringify([['audit', {}]]))}` };
  t.is(runTreeFromContext(tuple)?.replicas?.[0]?.projectName, 'audit');
  parent.replicas = [];
  t.deepEqual(runTreeFromContext(runHeaders(parent))?.replicas, []);
});

test('replicas use the receiving worker credentials for the matching destination', (t) => {
  t.deepEqual(bindReplicaCredentials([{ projectName: 'secondary', primary: false }], [{
    projectName: 'secondary', apiKey: 'worker-key', apiUrl: 'https://worker.invalid', workspaceId: 'workspace',
  }]), [{
    projectName: 'secondary', primary: false, apiKey: 'worker-key', apiUrl: 'https://worker.invalid',
    workspaceId: 'workspace', client: undefined,
  }]);
});

function historyText(value: unknown): string {
  if (value instanceof Uint8Array) return Buffer.from(value).toString('utf8');
  if (value && typeof value === 'object') {
    return Object.entries(value).map(([key, item]) => `${key}:${historyText(item)}`).join('\n');
  }
  return String(value);
}

test.serial('environment replicas route unparented workflows using worker-local credentials', async (t) => {
  const previous = process.env.LANGSMITH_RUNS_ENDPOINTS;
  process.env.LANGSMITH_RUNS_ENDPOINTS = JSON.stringify([
    { api_url: 'https://primary.invalid', api_key: 'worker-primary-secret', project_name: 'primary', primary: true },
    { api_url: 'https://secondary.invalid', api_key: 'worker-secondary-secret', project_name: 'secondary' },
  ]);
  const collector = new InMemoryRunCollector();
  const create = collector.createRun;
  const calls: { name: unknown; apiKey: string | undefined }[] = [];
  collector.createRun = async (run, options?: { apiKey?: string }): Promise<void> => {
    calls.push({ name: run.name, apiKey: options?.apiKey });
    return create(run);
  };
  try {
    await withTracingWorker({
      collector,
      options: { addTemporalRuns: false, projectName: 'env-replica-source' },
      activities,
      workerOptions: { workflowsPath: WORKFLOWS_PATH, nexusServices: [comprehensiveNexusServiceHandler] },
      body: async ({ client, taskQueue, env }) => {
        const endpoint = await env.createNexusEndpoint(ROUTING_ENDPOINT, taskQueue);
        t.teardown(() => env.deleteNexusEndpoint(endpoint));
        const handle = await client.workflow.start(RoutingWorkflow, {
          taskQueue, workflowId: `env-replicas-${randomUUID()}`, args: ['plugin'],
        });
        t.is(await handle.result(), 'done');
        const history = historyText(await handle.fetchHistory());
        t.false(history.includes('worker-primary-secret'));
        t.false(history.includes('worker-secondary-secret'));
        for (const key of ['worker-primary-secret', 'worker-secondary-secret']) {
          t.true(calls.some((call) => call.name === 'routing-step' && call.apiKey === key));
        }
        t.true(collector.records.some((run) => run.name === 'nexus_inner_call' && run.project_name === 'primary'));
        t.is(collector.records.filter((run) => run.project_name === 'primary').length * 2, collector.records.length);
        t.true(collector.records.every((run) => run.end_time !== undefined));
      },
    });
  } finally {
    if (previous === undefined) delete process.env.LANGSMITH_RUNS_ENDPOINTS;
    else process.env.LANGSMITH_RUNS_ENDPOINTS = previous;
  }
});

for (const addTemporalRuns of [false, true]) {
  for (const source of ['plugin', 'decorator', 'runtime', 'inherited']) {
    test.serial(`agent destination survives workflow, activity, child and Nexus (${source}, ${addTemporalRuns})`, async (t) => {
      const collector = new InMemoryRunCollector();
      await withTracingWorker({
        collector,
        options: {
          addTemporalRuns,
          ...(source === 'plugin' ? { address: ROUTING_ADDRESS } : { projectName: 'fallback' }),
        },
        activities,
        workerOptions: {
          workflowsPath: WORKFLOWS_PATH,
          maxCachedWorkflows: 0,
          nexusServices: [comprehensiveNexusServiceHandler],
        },
        body: async ({ client, taskQueue, env }) => {
          const endpoint = await env.createNexusEndpoint(ROUTING_ENDPOINT, taskQueue);
          t.teardown(() => env.deleteNexusEndpoint(endpoint));
          const execute = async (): Promise<void> => {
            const handle = await client.workflow.start(RoutingWorkflow, {
              taskQueue, workflowId: `routing-${randomUUID()}`, args: [source],
            });
            t.is(await handle.result(), 'done');
            const history = await handle.fetchHistory();
            t.true(historyText(history).includes(HEADER_KEY));
            const replay = new InMemoryRunCollector();
            await Worker.runReplayHistory({
              workflowsPath: WORKFLOWS_PATH,
              plugins: [new LangSmithPlugin({ client: replay.asClient(), addTemporalRuns, address: ROUTING_ADDRESS })],
            }, history, handle.workflowId);
            await Worker.runReplayHistory({
              workflowsPath: WORKFLOWS_PATH,
              plugins: [new LangSmithPlugin({ client: replay.asClient(), addTemporalRuns, address: ROUTING_ADDRESS })],
            }, await env.client.workflow.getHandle(`${handle.workflowId}-child`).fetchHistory(), `${handle.workflowId}-child`);
            t.deepEqual(replay.records, []);
          };
          if (source === 'inherited') {
            await traceable(execute, {
              name: 'client-root', address: ROUTING_ADDRESS, client: collector.asClient(), tracingEnabled: true,
            })();
          } else {
            await execute();
          }
          for (const name of ['routing-step', 'routing-activity', 'routing-child', 'nexus_inner_call']) {
            const runs = collector.records.filter((run) => run.name === name);
            t.true(runs.length > 0, `missing ${name}`);
            // An emitted RunWorkflow is a real parent, so its project outranks native child options.
            const expectedAddress = addTemporalRuns && (source === 'decorator' || source === 'runtime')
              ? undefined : ROUTING_ADDRESS;
            for (const run of runs) {
              t.is(run.address, expectedAddress, name);
              t.is(run.project_name, expectedAddress ? undefined : 'fallback', name);
              t.truthy(run.end_time, name);
            }
          }
          for (const run of collector.records.filter((item) => item.parent_run_id)) {
            const parent = collector.byId.get(run.parent_run_id!);
            t.truthy(parent, `${run.name} has a real parent`);
            t.true(run.dotted_order!.startsWith(`${parent!.dotted_order}.`));
          }
        },
      });
    });
  }

  test.serial(`primary and secondary project/agent replicas keep matching IDs and updates (${addTemporalRuns})`, async (t) => {
    const collector = new InMemoryRunCollector();
    const previous = process.env.LANGSMITH_RUNS_ENDPOINTS;
    process.env.LANGSMITH_RUNS_ENDPOINTS = JSON.stringify([
      { api_url: 'https://primary.invalid', api_key: 'worker-primary-secret', project_name: 'primary', primary: true },
      { api_url: 'https://secondary.invalid', api_key: 'worker-secondary-secret', project_name: 'secondary' },
    ]);
    try {
      await withTracingWorker({
        collector,
        options: { addTemporalRuns },
        activities,
        workerOptions: { workflowsPath: WORKFLOWS_PATH, maxCachedWorkflows: 0, nexusServices: [comprehensiveNexusServiceHandler] },
        body: async ({ client, taskQueue, env }) => {
          const endpoint = await env.createNexusEndpoint(ROUTING_ENDPOINT, taskQueue);
          t.teardown(() => env.deleteNexusEndpoint(endpoint));
          let originalRootId: string | undefined;
          await traceable(async () => {
            const handle = await client.workflow.start(RoutingWorkflow, {
              taskQueue, workflowId: `replicas-${randomUUID()}`, args: ['inherited'],
            });
            t.is(await handle.result(), 'done');
            const history = await handle.fetchHistory();
            const text = historyText(history);
            t.false(text.includes('worker-primary-secret'));
            t.false(text.includes('worker-secondary-secret'));
            const replay = new InMemoryRunCollector();
            await Worker.runReplayHistory({
              workflowsPath: WORKFLOWS_PATH,
              plugins: [new LangSmithPlugin({ client: replay.asClient(), addTemporalRuns })],
            }, history, handle.workflowId);
            await Worker.runReplayHistory({
              workflowsPath: WORKFLOWS_PATH,
              plugins: [new LangSmithPlugin({ client: replay.asClient(), addTemporalRuns })],
            }, await env.client.workflow.getHandle(`${handle.workflowId}-child`).fetchHistory(), `${handle.workflowId}-child`);
            t.deepEqual(replay.records, []);
          }, {
            name: 'client-root', client: collector.asClient(), tracingEnabled: true, replicas: REPLICAS,
            on_end: (run) => { originalRootId = run.id; },
          })();
          const primary = collector.records.filter((run) => run.project_name === 'primary');
          t.true(primary.some((run) => run.name === 'nexus_inner_call'));
          t.is(collector.byName('client-root')?.id, originalRootId, 'primary preserves the original root ID');
          for (const run of primary) {
            t.is(run.trace_id, originalRootId, 'primary preserves the original trace ID');
            // LangSmith's public ID helper accepts only UUIDv7; Workflow IDs are UUIDv4.
            // Use its actual serializer as the reference for both kinds of ID.
            const expected = new InMemoryRunCollector();
            await new RunTree({
              name: run.name,
              id: run.id,
              trace_id: run.trace_id,
              parent_run_id: run.parent_run_id,
              dotted_order: run.dotted_order,
              start_time: run.start_time,
              client: expected.asClient(),
              replicas: [{ projectName: 'secondary', primary: false }, { address: SECONDARY_ADDRESS, primary: false }],
            }).postRun();
            for (const reference of expected.records) {
              const secondary = collector.byId.get(reference.id);
              t.truthy(secondary, `${run.name} replicated to ${reference.session_name ?? reference.address}`);
              t.truthy(secondary?.end_time, `${run.name} replica completed`);
              t.is(secondary?.parent_run_id, reference.parent_run_id);
              t.is(secondary?.trace_id, reference.trace_id);
              t.is(secondary?.dotted_order, reference.dotted_order);
              if (reference.session_name === 'secondary') t.deepEqual(secondary?.tags, ['secondary-only']);
              if (reference.address) t.deepEqual(secondary?.extra?.metadata, { deployment: 'production' });
            }
          }
          t.is(collector.records.length, primary.length * 3);
        },
      });
    } finally {
      if (previous === undefined) delete process.env.LANGSMITH_RUNS_ENDPOINTS;
      else process.env.LANGSMITH_RUNS_ENDPOINTS = previous;
    }
  });
}
