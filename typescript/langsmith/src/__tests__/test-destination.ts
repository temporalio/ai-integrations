/**
 * Destination end-to-end. With no plugin project, runs go where LangSmith's env defaults send them unless a client-side
 * or workflow-body `traceable` names its own project or address, and a misconfigured agent env traces nothing. A plugin
 * project outranks a parent's address on runs built outside the workflow isolate. The plugin's client is a real
 * LangSmith `Client` over a capturing `fetch`, so LangSmith's own destination rules run. Some tests load LangSmith's
 * ESM build for the app side, as an ESM app does, beside the plugin's CommonJS build.
 *
 * @module
 */

import { randomUUID } from 'crypto';

import test from 'ava';
import { Client } from 'langsmith';
import { RunTree } from 'langsmith/run_trees';
import { traceable } from 'langsmith/traceable';

import { createClientInterceptor } from '../client-interceptor';
import * as activities from './activities/comprehensive';
import { useSharedEnv, withTracingWorker } from './helpers';
import { comprehensiveNexusServiceHandler } from './stubs/nexus';
import { DestinationWorkflow, OwnDestinationWorkflow } from './workflows/comprehensive';

process.env.LANGSMITH_TRACING = 'true';

const DESTINATION_ENV_VARS = ['LANGSMITH_PROJECT', 'LANGSMITH_AGENT_ID', 'LANGSMITH_AGENT_ENVIRONMENT'];

const AGENT_ENV = { LANGSMITH_AGENT_ID: 'support', LANGSMITH_AGENT_ENVIRONMENT: 'staging' };
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

/** `AGENT_ADDRESS` in the installed LangSmith's own address form. */
function installedAddress(build: LangSmithBuild = CJS_BUILD): RunTree['address'] {
  return build.RunTree.fromHeaders({
    'langsmith-trace': `20240101T000000000000Z${randomUUID()}`,
    baggage: `langsmith-address=${encodeURIComponent(AGENT_ADDRESS)}`,
  })?.address;
}

interface CreatedRun {
  name: string;
  trace_id: string;
  destination?: string;
}

/** Run `fn` with `env` as the only destination env vars, restoring them afterwards. */
async function withDestinationEnv<T>(env: Record<string, string>, fn: () => Promise<T>): Promise<T> {
  const saved = DESTINATION_ENV_VARS.map((name) => [name, process.env[name]] as const);
  for (const name of DESTINATION_ENV_VARS) {
    delete process.env[name];
  }
  Object.assign(process.env, env);
  try {
    return await fn();
  } finally {
    for (const [name, value] of saved) {
      if (value === undefined) {
        delete process.env[name];
      } else {
        process.env[name] = value;
      }
    }
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
  clientOptions: { address?: RunTree['address'] } = {},
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
                args: [OTHER_ADDRESS],
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
  clientOptions: { address?: RunTree['address'] } = {},
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
  'client traceable with an address and a plugin projectName: runs built outside the isolate go to the plugin project',
  async (t) => {
    for (const [addTemporalRuns, pluginProjectRuns] of [
      [true, ['StartWorkflow:DestinationWorkflow']],
      [false, ['comprehensive_activity', 'comprehensive_activity_inner', 'nexus_inner_call']],
    ] as const) {
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
      const destinationsOf = (name: string): (string | undefined)[] => [
        ...new Set(runs.filter((run) => run.name === name).map((run) => run.destination)),
      ];
      t.deepEqual(destinationsOf('user_pipeline'), [AGENT_ADDRESS]);
      for (const name of pluginProjectRuns) {
        t.deepEqual(destinationsOf(name), ['plugin-project'], name);
      }
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

// No client-side run, so the client interceptor's own `StartWorkflow:` marker is built too.
test.serial('misconfigured agent env vars: workflows complete and no run is created or updated', async (t) => {
  const { result, runs, updates } = await runWithEnv({ LANGSMITH_AGENT_ID: 'support' }, true, false);
  t.is(result, 'wf:d');
  t.deepEqual(runs, []);
  t.deepEqual(updates, []);
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
