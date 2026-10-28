/**
 * Destination end-to-end. With no plugin project, runs go where LangSmith's env defaults send them unless a client-side
 * or workflow-body `traceable` names its own project or address. A misconfigured agent env traces nothing that starts
 * without a parent, while runs below a parent that names a destination still join it, ahead of a plugin project.
 * The plugin's client is a real LangSmith `Client` over a
 * capturing `fetch`, so LangSmith's own destination rules run. Some tests load LangSmith's ESM build for the app side,
 * as an ESM app does, beside the plugin's CommonJS build.
 *
 * @module
 */

import { randomUUID } from 'crypto';

import test from 'ava';
import { Client } from 'langsmith';
import { RunTree } from 'langsmith/run_trees';
import { traceable } from 'langsmith/traceable';

import type { Payload } from '@temporalio/common';
import { Worker } from '@temporalio/worker';

import { createClientInterceptor } from '../client-interceptor';
import { LangSmithPlugin, envDestination } from '../plugin';
import { readContextHeader } from '../propagation';
import * as activities from './activities/comprehensive';
import { useSharedEnv, withTracingWorker } from './helpers';
import { comprehensiveNexusServiceHandler } from './stubs/nexus';
import {
  ActivityTimerWorkflow,
  DestinationWorkflow,
  OwnDestinationWorkflow,
  SignalDestinationWorkflow,
  destinationSignal,
} from './workflows/comprehensive';

process.env.LANGSMITH_TRACING = 'true';

const DESTINATION_ENV_VARS = ['LANGSMITH_PROJECT', 'LANGSMITH_AGENT_ID', 'LANGSMITH_AGENT_ENVIRONMENT'];

const AGENT_ENV = { LANGSMITH_AGENT_ID: 'support', LANGSMITH_AGENT_ENVIRONMENT: 'staging' };
const HALF_AGENT_ENV = { LANGSMITH_AGENT_ID: 'support' };
const OTHER_AGENT_ENV = { LANGSMITH_AGENT_ID: 'other', LANGSMITH_AGENT_ENVIRONMENT: 'production' };
const AGENT_ADDRESS = 'lrn:agents/support/environments/staging';
const OTHER_ADDRESS = 'lrn:agents/other/environments/production';

const WORKFLOWS_PATH = require.resolve('./workflows/comprehensive');

// tsc's CommonJS output turns import() into require(), which loads LangSmith's CommonJS build.
const esmImport = new Function('specifier', 'return import(specifier)') as <T>(specifier: string) => Promise<T>;

/** The parts of one LangSmith build (ESM or CommonJS) the tests use. */
interface LangSmithBuild {
  Client: typeof Client;
  RunTree: typeof RunTree;
  traceable: typeof traceable;
}

const CJS_BUILD: LangSmithBuild = { Client, RunTree, traceable };

let esmBuild: Promise<LangSmithBuild> | undefined;

/** LangSmith's ESM build, the copy an ESM app loads. */
function loadEsmBuild(): Promise<LangSmithBuild> {
  esmBuild ??= Promise.all([
    esmImport<{ Client: typeof Client }>('langsmith'),
    esmImport<{ RunTree: typeof RunTree }>('langsmith/run_trees'),
    esmImport<{ traceable: typeof traceable }>('langsmith/traceable'),
  ]).then(([langsmith, runTrees, traceables]) => ({
    Client: langsmith.Client,
    RunTree: runTrees.RunTree,
    traceable: traceables.traceable,
  }));
  return esmBuild;
}

interface Builds {
  app: LangSmithBuild;
  plugin: LangSmithBuild;
}

/** Destination options for the client-side `traceable`. */
interface ClientRunOptions {
  address?: RunTree['address'];
  project_name?: string;
}

interface CreatedRun {
  name: string;
  trace_id: string;
  destination?: string;
}

/** Set `env` as the only destination env vars, returning a function that restores the previous ones. */
function setDestinationEnv(env: Record<string, string>): () => void {
  const saved = DESTINATION_ENV_VARS.map((name) => [name, process.env[name]] as const);
  for (const name of DESTINATION_ENV_VARS) {
    delete process.env[name];
  }
  Object.assign(process.env, env);
  return () => {
    for (const [name, value] of saved) {
      if (value === undefined) {
        delete process.env[name];
      } else {
        process.env[name] = value;
      }
    }
  };
}

/** Run `fn` with `env` as the only destination env vars, restoring them afterwards. */
async function withDestinationEnv<T>(env: Record<string, string>, fn: () => Promise<T>): Promise<T> {
  const restore = setDestinationEnv(env);
  try {
    return await fn();
  } finally {
    restore();
  }
}

/**
 * The agent address `env` names (`AGENT_ADDRESS` by default) in `build`'s own address form, as LangSmith's `RunTree`
 * constructor resolves it for a run that names no destination. The probe run is never posted.
 */
function installedAddress(
  build: LangSmithBuild = CJS_BUILD,
  env: Record<string, string> = AGENT_ENV
): RunTree['address'] {
  const restore = setDestinationEnv(env);
  try {
    const client = new build.Client({ apiUrl: 'http://langsmith.test', apiKey: 'test' });
    return new build.RunTree({ name: 'address-probe', client, tracingEnabled: false }).address;
  } finally {
    restore();
  }
}

/** A `Client` from `build` over a `fetch` that records created runs in `runs` and updated run names in `updates`. */
function capturingClient(build: LangSmithBuild, runs: CreatedRun[], updates: string[]): Client {
  return new build.Client({
    apiUrl: 'http://langsmith.test',
    apiKey: 'test',
    fetchImplementation: async (input, init) => {
      const url = String(input);
      if (url.endsWith('/info')) {
        return new Response(JSON.stringify({ batch_ingest_config: { use_multipart_endpoint: false } }));
      }
      if (url.endsWith('/runs/batch')) {
        const body = JSON.parse(await new Response(init?.body).text()) as {
          post?: Record<string, string>[];
          patch?: Record<string, string>[];
        };
        for (const run of body.post ?? []) {
          runs.push({ name: run.name!, trace_id: run.trace_id!, destination: run.address ?? run.session_name });
        }
        for (const run of body.patch ?? []) {
          updates.push(run.name ?? run.id!);
        }
      }
      return new Response('{}');
    },
  });
}

/**
 * Run `DestinationWorkflow` (or `OwnDestinationWorkflow` when `ownDestination`) with `env` as the only destination env
 * vars, under a client-side `traceable` built with `clientOptions` when `clientRun`, returning its result and the runs the
 * LangSmith clients created and updated. `builds` picks the LangSmith build (ESM or CommonJS) the app side and the
 * plugin's client come from, and `projectName` is the plugin's `projectName` option.
 */
async function runWithEnv(
  env: Record<string, string>,
  addTemporalRuns: boolean,
  clientRun: boolean,
  clientOptions: ClientRunOptions = {},
  ownDestination = false,
  builds: Builds = { app: CJS_BUILD, plugin: CJS_BUILD },
  projectName?: string
): Promise<{ result: string; runs: CreatedRun[]; updates: string[] }> {
  return withDestinationEnv(env, async () => {
    const runs: CreatedRun[] = [];
    const updates: string[] = [];
    const appClient = capturingClient(builds.app, runs, updates);
    const pluginClient = builds.plugin === builds.app ? appClient : capturingClient(builds.plugin, runs, updates);
    let result = '';
    await withTracingWorker({
      collector: { asClient: () => pluginClient },
      options: { addTemporalRuns, projectName },
      activities: { comprehensiveActivity: activities.comprehensiveActivity },
      workerOptions: { workflowsPath: WORKFLOWS_PATH, nexusServices: [comprehensiveNexusServiceHandler] },
      body: async ({ client, taskQueue, env: testEnv }) => {
        const endpoint = `destination-${randomUUID()}`;
        await testEnv.createNexusEndpoint(endpoint, taskQueue);
        const execute = async (): Promise<string> =>
          ownDestination
            ? client.workflow.execute(OwnDestinationWorkflow, {
                taskQueue,
                workflowId: endpoint,
                args: [installedAddress(CJS_BUILD, OTHER_AGENT_ENV)],
                workflowExecutionTimeout: '30s',
              })
            : client.workflow.execute(DestinationWorkflow, { taskQueue, workflowId: endpoint, args: [endpoint] });
        result = clientRun
          ? await builds.app.traceable(execute, { name: 'user_pipeline', client: appClient, ...clientOptions })()
          : await execute();
      },
    });
    await appClient.awaitPendingTraceBatches();
    await pluginClient.awaitPendingTraceBatches();
    return { result, runs, updates };
  });
}

const USER_RUNS = [
  'user_pipeline',
  'comprehensive_activity',
  'comprehensive_activity_inner',
  'nexus_inner_call',
  'workflow_inner_call',
];

const TEMPORAL_RUNS = [
  'StartWorkflow:DestinationWorkflow',
  'RunWorkflow:DestinationWorkflow',
  'StartActivity:comprehensiveActivity',
  'RunActivity:comprehensiveActivity',
  'StartChildWorkflow:ComprehensiveChildWorkflow',
  'RunWorkflow:ComprehensiveChildWorkflow',
  'StartNexusOperation:comprehensiveNexusService/greet',
  'RunStartNexusOperationHandler:comprehensiveNexusService/greet',
];

/** Asserts the workflow completes, every run is created, and its one trace goes only to `destination`. */
async function assertEveryRunGoesTo(
  t: import('ava').ExecutionContext,
  env: Record<string, string>,
  addTemporalRuns: boolean,
  destination: string,
  clientOptions: ClientRunOptions = {},
  builds?: Builds
): Promise<void> {
  const { result, runs } = await runWithEnv(env, addTemporalRuns, true, clientOptions, false, builds);
  t.is(result, 'wf:d');
  t.deepEqual(
    [...new Set(runs.map((run) => run.name))].sort(),
    [...USER_RUNS, ...(addTemporalRuns ? TEMPORAL_RUNS : [])].sort()
  );
  const destinationsByTrace = new Map<string, Set<string | undefined>>();
  for (const run of runs) {
    destinationsByTrace.set(run.trace_id, (destinationsByTrace.get(run.trace_id) ?? new Set()).add(run.destination));
  }
  t.deepEqual(
    [...destinationsByTrace.values()].map((destinations) => [...destinations]),
    [[destination]]
  );
}

useSharedEnv(test);

test.serial('agent env vars: every run goes to the agent address (addTemporalRuns=true)', async (t) => {
  await assertEveryRunGoesTo(t, AGENT_ENV, true, AGENT_ADDRESS);
});

test.serial('agent env vars: every run goes to the agent address (addTemporalRuns=false)', async (t) => {
  await assertEveryRunGoesTo(t, AGENT_ENV, false, AGENT_ADDRESS);
});

test.serial('LANGSMITH_PROJECT: every run goes to the env project', async (t) => {
  await assertEveryRunGoesTo(t, { LANGSMITH_PROJECT: 'env-project' }, true, 'env-project');
});

test.serial('client traceable with an address: every run goes to that address (addTemporalRuns=false)', async (t) => {
  await assertEveryRunGoesTo(t, { LANGSMITH_PROJECT: 'env-project' }, false, AGENT_ADDRESS, {
    address: installedAddress(),
  });
});

test.serial(
  'client traceable with an address: the StartWorkflow marker goes to that address (addTemporalRuns=true)',
  async (t) => {
    const { result, runs } = await runWithEnv({ LANGSMITH_PROJECT: 'env-project' }, true, true, {
      address: installedAddress(),
    });
    t.is(result, 'wf:d');
    for (const name of ['user_pipeline', 'StartWorkflow:DestinationWorkflow']) {
      t.deepEqual([...new Set(runs.filter((run) => run.name === name).map((run) => run.destination))], [AGENT_ADDRESS]);
    }
  }
);

test.serial(
  'client traceable with an address and a plugin projectName: every run inherits the parent address',
  async (t) => {
    for (const addTemporalRuns of [true, false]) {
      const { result, runs } = await runWithEnv(
        { LANGSMITH_PROJECT: 'env-project' },
        addTemporalRuns,
        true,
        { address: installedAddress() },
        false,
        undefined,
        'plugin-project'
      );
      t.is(result, 'wf:d');
      t.deepEqual([...new Set(runs.map((run) => run.destination))], [AGENT_ADDRESS]);
    }
  }
);

test.serial(
  'workflow-body traceables naming their own project or address keep it under a parent (addTemporalRuns=false)',
  async (t) => {
    for (const [clientOptions, parentDestination] of [
      [{}, 'env-project'],
      [{ address: installedAddress() }, AGENT_ADDRESS],
    ] as const) {
      const { result, runs } = await runWithEnv({ LANGSMITH_PROJECT: 'env-project' }, false, true, clientOptions, true);
      t.is(result, 'pawf:d');
      t.deepEqual(Object.fromEntries(runs.map((run) => [run.name, run.destination])), {
        user_pipeline: parentDestination,
        own_project_call: 'own-project',
        own_address_call: OTHER_ADDRESS,
        workflow_inner_call: parentDestination,
      });
    }
  }
);

// No client run: the workflow-body traceable starts its own trace, so only destinations are compared.
test.serial(
  'no client run, LANGSMITH_PROJECT: every run goes to the env project (addTemporalRuns=false)',
  async (t) => {
    const { result, runs } = await runWithEnv({ LANGSMITH_PROJECT: 'env-project' }, false, false);
    t.is(result, 'wf:d');
    t.deepEqual(
      [...new Set(runs.map((run) => run.name))].sort(),
      USER_RUNS.filter((name) => name !== 'user_pipeline').sort()
    );
    t.deepEqual([...new Set(runs.map((run) => run.destination))], ['env-project']);
  }
);

test.serial(
  'no client run, LANGSMITH_PROJECT: workflow-body traceables naming their own project or address keep it',
  async (t) => {
    const { result, runs } = await runWithEnv({ LANGSMITH_PROJECT: 'env-project' }, false, false, {}, true);
    t.is(result, 'pawf:d');
    t.deepEqual(Object.fromEntries(runs.map((run) => [run.name, run.destination])), {
      own_project_call: 'own-project',
      own_address_call: OTHER_ADDRESS,
      workflow_inner_call: 'env-project',
    });
  }
);

test.serial(
  'no client run, plugin projectName: every run goes to the plugin project (addTemporalRuns=false)',
  async (t) => {
    const { result, runs } = await runWithEnv({}, false, false, {}, false, undefined, 'plugin-project');
    t.is(result, 'wf:d');
    t.deepEqual(
      [...new Set(runs.map((run) => run.name))].sort(),
      USER_RUNS.filter((name) => name !== 'user_pipeline').sort()
    );
    t.deepEqual([...new Set(runs.map((run) => run.destination))], ['plugin-project']);
  }
);

// No client-side run, so the client interceptor's own `StartWorkflow:` marker is built too.
test.serial('misconfigured agent env vars: workflows complete and no run is created or updated', async (t) => {
  const { result, runs, updates } = await runWithEnv(HALF_AGENT_ENV, true, false);
  t.is(result, 'wf:d');
  t.deepEqual(runs, []);
  t.deepEqual(updates, []);
});

test.serial(
  'misconfigured agent env vars: a signal handler traceable with no parent completes and creates no run',
  async (t) => {
    await withDestinationEnv(HALF_AGENT_ENV, async () => {
      const runs: CreatedRun[] = [];
      const updates: string[] = [];
      const pluginClient = capturingClient(CJS_BUILD, runs, updates);
      await withTracingWorker({
        collector: { asClient: () => pluginClient },
        options: { addTemporalRuns: true },
        activities: {},
        workerOptions: { workflowsPath: WORKFLOWS_PATH },
        body: async ({ client, taskQueue }) => {
          const handle = await client.workflow.start(SignalDestinationWorkflow, {
            taskQueue,
            workflowId: `signal-destination-${randomUUID()}`,
          });
          await handle.signal(destinationSignal);
          t.is(await handle.result(), 'signalled');
        },
      });
      await pluginClient.awaitPendingTraceBatches();
      t.deepEqual(runs, []);
      t.deepEqual(updates, []);
    });
  }
);

/**
 * Calls, with no ambient run, each client method that builds its own marker run and propagates it as the parent,
 * returning the headers each passed on.
 */
async function markerCallHeaders(
  interceptor: ReturnType<typeof createClientInterceptor>
): Promise<Record<string, Record<string, Payload>>> {
  const sent: Record<string, Record<string, Payload>> = {};
  const next = (method: string) =>
    (async (input: { headers?: Record<string, Payload>; workflowStartHeaders?: Record<string, Payload> }) => {
      sent[method] = (input.headers ?? input.workflowStartHeaders)!;
    }) as never;
  await interceptor.signal!({ signalName: 'sig', args: [], headers: {} } as never, next('signal'));
  await interceptor.signalWithStart!(
    { workflowType: 'Wf', signalArgs: [], headers: {} } as never,
    next('signalWithStart')
  );
  await interceptor.query!({ queryType: 'q', args: [], headers: {} } as never, next('query'));
  await interceptor.startUpdate!({ updateName: 'u', args: [], headers: {} } as never, next('startUpdate'));
  await interceptor.startUpdateWithStart!(
    { updateName: 'uws', updateArgs: [], workflowStartHeaders: {} } as never,
    next('startUpdateWithStart')
  );
  return sent;
}

test.serial('misconfigured agent env vars: a client marker LangSmith left untraced is not propagated', async (t) => {
  await withDestinationEnv(HALF_AGENT_ENV, async () => {
    const interceptor = createClientInterceptor({ client: capturingClient(CJS_BUILD, [], []), addTemporalRuns: true });
    const sent = await markerCallHeaders(interceptor);
    t.deepEqual(Object.keys(sent).sort(), [
      'query',
      'signal',
      'signalWithStart',
      'startUpdate',
      'startUpdateWithStart',
    ]);
    for (const [method, headers] of Object.entries(sent)) {
      t.is(readContextHeader(headers), undefined, method);
    }
  });
});

// The Temporal runs a Worker builds outside the workflow isolate, which follow an incoming parent.
const OUTSIDE_ISOLATE_TEMPORAL_RUNS = [
  'StartWorkflow:DestinationWorkflow',
  'RunActivity:comprehensiveActivity',
  'RunStartNexusOperationHandler:comprehensiveNexusService/greet',
];

for (const addTemporalRuns of [true, false]) {
  test.serial(
    `misconfigured agent env vars under a client run naming a project: runs join it (addTemporalRuns=${addTemporalRuns})`,
    async (t) => {
      const { result, runs } = await runWithEnv(HALF_AGENT_ENV, addTemporalRuns, true, {
        project_name: 'caller-project',
      });
      t.is(result, 'wf:d');
      t.deepEqual(
        [...new Set(runs.map((run) => run.name))].sort(),
        [...USER_RUNS, ...(addTemporalRuns ? OUTSIDE_ISOLATE_TEMPORAL_RUNS : [])].sort()
      );
      t.is(new Set(runs.map((run) => run.trace_id)).size, 1);
      t.deepEqual([...new Set(runs.map((run) => run.destination))], ['caller-project']);
    }
  );
}

test.serial(
  'empty projectName, misconfigured agent env vars: workflows complete and no run is created or updated',
  async (t) => {
    const { result, runs, updates } = await runWithEnv(HALF_AGENT_ENV, true, false, {}, false, undefined, '');
    t.is(result, 'wf:d');
    t.deepEqual(runs, []);
    t.deepEqual(updates, []);
  }
);

for (const addTemporalRuns of [true, false]) {
  test.serial(
    `empty projectName, misconfigured agent env vars under a client run naming a project: runs join it (addTemporalRuns=${addTemporalRuns})`,
    async (t) => {
      const { result, runs } = await runWithEnv(
        HALF_AGENT_ENV,
        addTemporalRuns,
        true,
        { project_name: 'caller-project' },
        false,
        undefined,
        ''
      );
      t.is(result, 'wf:d');
      t.deepEqual(
        [...new Set(runs.map((run) => run.name))].sort(),
        [...USER_RUNS, ...(addTemporalRuns ? OUTSIDE_ISOLATE_TEMPORAL_RUNS : [])].sort()
      );
      t.is(new Set(runs.map((run) => run.trace_id)).size, 1);
      t.deepEqual([...new Set(runs.map((run) => run.destination))], ['caller-project']);
    }
  );
}

for (const [recordName, recordEnv, replayName, replayEnv] of [
  ['valid', AGENT_ENV, 'misconfigured', HALF_AGENT_ENV],
  ['misconfigured', HALF_AGENT_ENV, 'valid', AGENT_ENV],
] as const) {
  test.serial(
    `replay: a history recorded under ${recordName} agent env vars replays under ${replayName} ones (addTemporalRuns=true)`,
    async (t) => {
      const history = await withDestinationEnv(recordEnv, () =>
        withTracingWorker({
          collector: { asClient: () => capturingClient(CJS_BUILD, [], []) },
          options: { addTemporalRuns: true },
          activities: { comprehensiveActivity: activities.comprehensiveActivity },
          workerOptions: { workflowsPath: WORKFLOWS_PATH },
          body: async ({ client, taskQueue }) => {
            const handle = await client.workflow.start(ActivityTimerWorkflow, {
              taskQueue,
              workflowId: `activity-timer-${randomUUID()}`,
            });
            await handle.result();
            return handle.fetchHistory();
          },
        })
      );
      await withDestinationEnv(replayEnv, async () => {
        const plugin = new LangSmithPlugin({ addTemporalRuns: true, client: capturingClient(CJS_BUILD, [], []) });
        await t.notThrowsAsync(Worker.runReplayHistory({ workflowsPath: WORKFLOWS_PATH, plugins: [plugin] }, history));
      });
    }
  );
}

test.serial('envDestination: resolves the env default the way LangSmith does, without warning', async (t) => {
  const client = new Client({ apiUrl: 'http://langsmith.test', apiKey: 'test' });
  const warnings: unknown[] = [];
  const { warn, error } = console;
  console.warn = (...args: unknown[]) => warnings.push(args);
  console.error = (...args: unknown[]) => warnings.push(args);
  try {
    t.deepEqual(
      await withDestinationEnv({ LANGSMITH_PROJECT: 'env-project' }, async () => envDestination(client)),
      { project_name: 'env-project' }
    );
    t.deepEqual(await withDestinationEnv(AGENT_ENV, async () => envDestination(client)), {
      address: installedAddress(),
    });
    t.is(await withDestinationEnv(HALF_AGENT_ENV, async () => envDestination(client)), undefined);
  } finally {
    console.warn = warn;
    console.error = error;
  }
  t.deepEqual(warnings, []);
});

const NEXT = (async () => undefined) as never;

const CLIENT_METHOD_CALLS: Record<
  string,
  { marker: string; call: (interceptor: ReturnType<typeof createClientInterceptor>) => Promise<unknown> }
> = {
  start: {
    marker: 'StartWorkflow:StartWf',
    call: (interceptor) =>
      // eslint-disable-next-line @typescript-eslint/no-deprecated -- the plugin still intercepts the deprecated `start`.
      interceptor.start!({ workflowType: 'StartWf', headers: {}, options: { args: [] } } as never, NEXT),
  },
  startWithDetails: {
    marker: 'StartWorkflow:StartWithDetailsWf',
    call: (interceptor) =>
      interceptor.startWithDetails!(
        { workflowType: 'StartWithDetailsWf', headers: {}, options: { args: [] } } as never,
        NEXT
      ),
  },
  signal: {
    marker: 'SignalWorkflow:sig',
    call: (interceptor) => interceptor.signal!({ signalName: 'sig', args: [], headers: {} } as never, NEXT),
  },
  signalWithStart: {
    marker: 'SignalWithStartWorkflow:SignalWithStartWf',
    call: (interceptor) =>
      interceptor.signalWithStart!({ workflowType: 'SignalWithStartWf', signalArgs: [], headers: {} } as never, NEXT),
  },
  query: {
    marker: 'QueryWorkflow:q',
    call: (interceptor) => interceptor.query!({ queryType: 'q', args: [], headers: {} } as never, NEXT),
  },
  startUpdate: {
    marker: 'StartWorkflowUpdate:u',
    call: (interceptor) => interceptor.startUpdate!({ updateName: 'u', args: [], headers: {} } as never, NEXT),
  },
  startUpdateWithStart: {
    marker: 'StartUpdateWithStartWorkflow:uws',
    call: (interceptor) =>
      interceptor.startUpdateWithStart!({ updateName: 'uws', updateArgs: [], workflowStartHeaders: {} } as never, NEXT),
  },
};

/**
 * Asserts every client interceptor method, called from an ESM-build `traceable` (given an address when `withAddress`),
 * sends its marker run to `destination`.
 */
async function assertEveryClientMethodMarkerGoesTo(
  t: import('ava').ExecutionContext,
  env: Record<string, string>,
  withAddress: boolean,
  destination: string
): Promise<void> {
  const esm = await loadEsmBuild();
  t.not(esm.Client, Client);
  const clientOptions = withAddress ? { address: installedAddress(esm) } : {};
  await withDestinationEnv(env, async () => {
    const runs: CreatedRun[] = [];
    const updates: string[] = [];
    const appClient = capturingClient(esm, runs, updates);
    const pluginClient = capturingClient(CJS_BUILD, runs, updates);
    const interceptor = createClientInterceptor({ client: pluginClient, addTemporalRuns: true });
    for (const [method, { call }] of Object.entries(CLIENT_METHOD_CALLS)) {
      await t.notThrowsAsync(
        esm.traceable(() => call(interceptor), { name: `app_${method}`, client: appClient, ...clientOptions })(),
        method
      );
    }
    await appClient.awaitPendingTraceBatches();
    await pluginClient.awaitPendingTraceBatches();
    const markers = runs.filter((run) => !run.name.startsWith('app_'));
    t.deepEqual(
      markers.map((run) => run.name).sort(),
      Object.values(CLIENT_METHOD_CALLS)
        .map(({ marker }) => marker)
        .sort()
    );
    t.deepEqual([...new Set(markers.map((run) => run.destination))], [destination]);
  });
}

test.serial('ESM app, every client interceptor method: markers go to the env agent address', async (t) => {
  await assertEveryClientMethodMarkerGoesTo(t, AGENT_ENV, false, AGENT_ADDRESS);
});

test.serial(
  'ESM app, every client interceptor method: markers go to the explicit address of the app run',
  async (t) => {
    await assertEveryClientMethodMarkerGoesTo(t, { LANGSMITH_PROJECT: 'env-project' }, true, AGENT_ADDRESS);
  }
);

for (const addTemporalRuns of [true, false]) {
  test.serial(
    `ESM app passes its own Client to the plugin: every run goes to the agent address (addTemporalRuns=${addTemporalRuns})`,
    async (t) => {
      const esm = await loadEsmBuild();
      t.not(esm.Client, Client);
      await assertEveryRunGoesTo(t, AGENT_ENV, addTemporalRuns, AGENT_ADDRESS, {}, { app: esm, plugin: esm });
    }
  );
}

test.serial(
  'ESM app with a CommonJS Client for the plugin: every run goes to the agent address (addTemporalRuns=true)',
  async (t) => {
    const esm = await loadEsmBuild();
    t.not(esm.Client, Client);
    await assertEveryRunGoesTo(t, AGENT_ENV, true, AGENT_ADDRESS, {}, { app: esm, plugin: CJS_BUILD });
  }
);
