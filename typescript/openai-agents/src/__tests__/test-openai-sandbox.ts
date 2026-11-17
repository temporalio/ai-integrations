import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { setTracingDisabled, withTrace } from '@openai/agents-core';
import { UnixLocalSandboxClient } from '@openai/agents-core/sandbox/local';
import { WorkflowClient, type WorkflowHandle } from '@temporalio/client';
import type { temporal } from '@temporalio/proto';
import { SimplePlugin } from '@temporalio/plugin';
import { NativeConnection, Worker, type WorkerOptions } from '@temporalio/worker';
import { helpers } from '@temporalio/test-helpers';
import {
  OnHostSandboxClientProvider,
  OpenAIAgentsPlugin,
  OpenAIAgentsTraceClientInterceptor,
  SandboxClientProvider,
} from '..';
import { installAgentSdkCollector } from './helpers/openai-agents-spans';
import { makeTestFunction } from './helpers/test-fn';
import { FakeModelProvider, handoffResponse, textResponse, toolCallResponse } from './stubs/openai-agents';
import {
  FakeSandboxClient,
  FakeSandboxSession,
  LifecycleSandboxClient,
  RootMovingSandboxClient,
} from './stubs/sandbox-fakes';
import {
  approvalPausedQuery,
  approvalResumeSignal,
  customHandoffSandboxAgentWithoutConfigWorkflow,
  customHandoffSandboxToolWorkflow,
  sandboxAgentWorkflow,
  proceedSignal,
  sandboxApprovalSequenceWorkflow,
  sandboxApprovalResumeWorkflow,
  sandboxArchiveLimitsWorkflow,
  sandboxClientChangedWorkflow,
  sandboxContinueAsNewWorkflow,
  sandboxExecWorkflow,
  sandboxManifestResumeWorkflow,
  sandboxNestedApprovalWorkflow,
  sandboxNestedRootResumeWorkflow,
  sandboxNoModelWorkflow,
  sandboxOperationsWorkflow,
  sandboxResumeEnvironmentWorkflow,
  sandboxRootResumeWorkflow,
  sandboxSavedSessionResumeWorkflow,
  sandboxValidationWorkflow,
  sandboxWorkerEnvWorkflow,
  sessionCreatedQuery,
} from './workflows/openai-sandbox';

setTracingDisabled(false);

const test = makeTestFunction({
  workflowsPath: require.resolve('./workflows/openai-sandbox'),
  plugins: [new OpenAIAgentsPlugin({ modelProvider: new FakeModelProvider([]) })],
});

const PERMITTED_SPAN_DATA_KEYS = new Set(['sessionId', 'port', 'exitCode', 'byteLength', 'length', 'count']);

const MANIFEST_SECRET_VAR = 'OPENAI_AGENTS_TEST_MANIFEST_SECRET';
const MANIFEST_SECRET = 'sk-approval-resume-sentinel';
const OTHER_SECRET_VAR = 'OPENAI_AGENTS_TEST_OTHER_SECRET';
const APPROVAL_VAR = 'OPENAI_AGENTS_TEST_APPROVAL_VALUE';

/** The Task Queue of each Activity a history scheduled, by Activity type, in order. */
function scheduledQueues(history: temporal.api.history.v1.IHistory): Array<[string, string]> {
  return (history.events ?? []).flatMap((event) => {
    const scheduled = event.activityTaskScheduledEventAttributes;
    return scheduled ? [[scheduled.activityType!.name!, scheduled.taskQueue!.name!] as [string, string]] : [];
  });
}

/** The text of every byte field in `value`, such as the payloads of history events. */
function recordedText(value: unknown): string[] {
  if (value instanceof Uint8Array) return [Buffer.from(value).toString('utf8')];
  if (value === null || typeof value !== 'object') return [];
  return Object.values(value).flatMap(recordedText);
}

test('SandboxAgent run exercises the full sandbox lifecycle through Activities', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const session = new FakeSandboxSession();
  const client = new FakeSandboxClient(session);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([
          toolCallResponse('run_command', { cmd: 'echo hello' }),
          toolCallResponse('read_file', { path: '/workspace/test.txt' }),
          toolCallResponse('write_file', { path: '/workspace/out.txt', diff: 'hello' }),
          textResponse('Done.'),
        ]),
        activityOptions: { startToCloseTimeout: '30s' },
        sandboxClientProviders: [new SandboxClientProvider('fake', client)],
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(sandboxAgentWorkflow);
    t.is(result, 'Done.');
  });

  t.is(client.createCalls.length, 1, 'client.create() not called');
  t.is(session.startCalls, 1, 'session.start() not called');
  t.true(session.execCalls.length >= 1, 'session.execCommand() not called');
  t.is(session.execCalls[0]!.cmd, 'echo hello');
  t.true(session.readFileCalls.length >= 1, 'session.readFile() not called');
  t.deepEqual(session.editorOperations, ['create:/workspace/out.txt'], 'editor createFile not called');
  t.true(session.stopCalls >= 1, 'session.stop() not called');
  t.true(session.shutdownCalls >= 1, 'session.shutdown() not called');
  t.true(session.deleteCalls >= 1, 'session.delete() not called');
});

// Serial: a Worker installs its allowlist process-wide, so a Worker created alongside would switch it off.
test.serial('applyManifest updates the Workflow-side manifest and survives resume', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([]),
        sandboxClientProviders: [new SandboxClientProvider('fake', new FakeSandboxClient())],
        resolvableWorkerEnvVars: [MANIFEST_SECRET_VAR],
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(sandboxManifestResumeWorkflow);
    t.is(result, 'live=true persisted=true reference=true');
  });
});

test.serial('a SandboxAgent whose manifest carries a workerEnvValue survives an approval interruption', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const client = new FakeSandboxClient();
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([
          toolCallResponse('run_command', { cmd: 'echo hello' }),
          textResponse('Done.'),
          toolCallResponse('run_command', { cmd: 'echo hello' }),
          textResponse('Done.'),
        ]),
        activityOptions: { startToCloseTimeout: '30s' },
        sandboxClientProviders: [new SandboxClientProvider('fake', client)],
        resolvableWorkerEnvVars: [MANIFEST_SECRET_VAR],
      }),
    ],
  });

  process.env[MANIFEST_SECRET_VAR] = MANIFEST_SECRET;
  try {
    await worker.runUntil(async () => {
      // The approved sandbox tool writes the caller's context object, not a copy rebuilt from JSON.
      const live = await startWorkflow(sandboxApprovalResumeWorkflow);
      t.is(await live.result(), 'Done. writes=sandbox-tool');
      const serialized = await startWorkflow(sandboxApprovalResumeWorkflow, { args: [true] });
      t.regex(await serialized.result(), /^Done\./);
      for (const handle of [live, serialized]) {
        const recorded = recordedText((await handle.fetchHistory()).events).join('\n');
        t.true(recorded.includes(MANIFEST_SECRET_VAR), 'history records the Worker environment variable name');
        t.false(recorded.includes(MANIFEST_SECRET), 'history never records its value');
      }
    });
  } finally {
    delete process.env[MANIFEST_SECRET_VAR];
  }

  t.true(client.resumeCalls >= 1, 'the preserved session is re-established through resume()');
});

test.serial("a resumed sandbox gets the environment of the agent's current manifest, not of the paused state", async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const client = new FakeSandboxClient();
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([
          toolCallResponse('run_command', { cmd: 'echo hello' }),
          textResponse('Done.'),
          toolCallResponse('run_command', { cmd: 'echo hello' }),
          textResponse('Done.'),
        ]),
        activityOptions: { startToCloseTimeout: '30s' },
        sandboxClientProviders: [new SandboxClientProvider('fake', client)],
        resolvableWorkerEnvVars: [MANIFEST_SECRET_VAR, OTHER_SECRET_VAR],
      }),
    ],
  });

  process.env[MANIFEST_SECRET_VAR] = MANIFEST_SECRET;
  try {
    await worker.runUntil(async () => {
      t.is(await executeWorkflow(sandboxResumeEnvironmentWorkflow, { args: ['changed'] }), 'Done.');
      t.deepEqual(client.resumeEnvironments.at(-1), {
        API_KEY: { type: 'temporal.worker-env-value', name: OTHER_SECRET_VAR },
      });
      t.is(await executeWorkflow(sandboxResumeEnvironmentWorkflow, { args: ['extra-reference'] }), 'Done.');
      t.deepEqual(client.resumeEnvironments.at(-1), {
        API_KEY: { type: 'temporal.worker-env-value', name: MANIFEST_SECRET_VAR },
      });
    });
  } finally {
    delete process.env[MANIFEST_SECRET_VAR];
  }
});

for (const [copied, parentName, rebuilt] of [
  [false, 'sandbox-parent', false],
  [true, 'sandbox-parent', false],
  [false, 'Nested', false],
  [true, 'Nested', false],
  [false, 'sandbox-parent', true],
  [true, 'sandbox-parent', true],
  [false, 'Nested', true],
  [true, 'Nested', true],
] as const) {
  // Serial: each case starts a Worker, and starting many at once can starve a slow CI runner.
  test.serial(
    `A nested approval behind ${copied ? 'a copy of an' : 'an'} asTool() tool in a sandbox run whose parent is named '${parentName}' resumes${rebuilt ? ' with rebuilt agents' : ''}`,
    async (t) => {
      const { createWorker, executeWorkflow } = helpers(t);
      const worker = await createWorker({
        plugins: [
          new OpenAIAgentsPlugin({
            modelProvider: new FakeModelProvider([
              toolCallResponse('ask_nested', { input: 'q' }),
              toolCallResponse('confirmAction', { reason: 'approved' }),
              textResponse('nested done'),
              textResponse('parent done'),
            ]),
            sandboxClientProviders: [new SandboxClientProvider('fake', new FakeSandboxClient())],
          }),
        ],
      });

      await worker.runUntil(async () => {
        const result = await executeWorkflow(sandboxNestedApprovalWorkflow, { args: [copied, parentName, rebuilt] });
        t.is(result, 'parent done executions=1');
      });
    }
  );
}

for (const onHost of [false, true]) {
  // Serial: a Worker installs its allowlist process-wide, so a Worker created alongside would switch it off.
  test.serial(
    'Approval resumes with a new client keep the live sandbox and run it with the Worker variable the manifest ' +
      `names${onHost ? ' (on-host)' : ''}`,
    async (t) => {
      const { createWorker, executeWorkflow } = helpers(t);
      const backend = new LifecycleSandboxClient();
      const third = toolCallResponse('run_approved', { cmd: 'third' });
      Object.assign(third.output[0]!, { callId: 'call_fake_run_approved_third' });
      const worker = await createWorker({
        plugins: [
          new OpenAIAgentsPlugin({
            modelProvider: new FakeModelProvider([
              toolCallResponse('run_now', { cmd: 'first' }),
              toolCallResponse('run_approved', { cmd: 'second' }),
              third,
              textResponse('Done.'),
            ]),
            sandboxClientProviders: [
              onHost
                ? new OnHostSandboxClientProvider('lifecycle', backend, { workerName: 'approvals' })
                : new SandboxClientProvider('lifecycle', backend),
            ],
            resolvableWorkerEnvVars: [APPROVAL_VAR],
          }),
        ],
      });

      process.env[APPROVAL_VAR] = 'resolved-on-worker';
      try {
        const output = await worker.runUntil(
          executeWorkflow(sandboxApprovalSequenceWorkflow, { args: [[APPROVAL_VAR, APPROVAL_VAR, APPROVAL_VAR]] })
        );
        t.is(
          output,
          [
            '#1 ran first with KEY=resolved-on-worker',
            '#1 ran first,second with KEY=resolved-on-worker',
            '#1 ran first,second,third with KEY=resolved-on-worker',
          ].join('\n')
        );
        t.deepEqual(
          backend.events.filter((event) => /^(create|resume|delete)/.test(event)),
          ['create#1', 'delete#1']
        );
        t.deepEqual(backend.liveSandboxes(), []);
      } finally {
        delete process.env[APPROVAL_VAR];
      }
    }
  );
}

// Serial: a Worker installs its allowlist process-wide, so a Worker created alongside would switch it off.
test.serial('An approval resume on a Worker that lost the live sandbox resumes it there and keeps it', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const backend = new LifecycleSandboxClient();
  const workerWith = (responses: ConstructorParameters<typeof FakeModelProvider>[0]) =>
    createWorker({
      maxCachedWorkflows: 0,
      plugins: [
        new OpenAIAgentsPlugin({
          modelProvider: new FakeModelProvider(responses),
          sandboxClientProviders: [new SandboxClientProvider('lifecycle', backend)],
          resolvableWorkerEnvVars: [APPROVAL_VAR],
        }),
      ],
    });

  process.env[APPROVAL_VAR] = 'resolved-on-worker';
  try {
    const first = await workerWith([
      toolCallResponse('run_now', { cmd: 'first' }),
      toolCallResponse('run_approved', { cmd: 'second' }),
    ]);
    const handle = await first.runUntil(async () => {
      const handle = await startWorkflow(sandboxApprovalSequenceWorkflow, {
        args: [[APPROVAL_VAR, APPROVAL_VAR], true],
      });
      while (!(await handle.query(approvalPausedQuery))) await new Promise((resolve) => setTimeout(resolve, 100));
      return handle;
    });

    // A new Worker caches no sessions, as after a restart.
    const second = await workerWith([textResponse('Done.')]);
    const output = await second.runUntil(async () => {
      await handle.signal(approvalResumeSignal);
      return await handle.result();
    });

    t.is(output, '#1 ran first with KEY=resolved-on-worker\n#1 ran first,second with KEY=resolved-on-worker');
    t.deepEqual(
      backend.events.filter((event) => /^(create|resume|delete)/.test(event)),
      ['create#1', 'resume#1', 'delete#1']
    );
    t.deepEqual(backend.liveSandboxes(), []);
  } finally {
    delete process.env[APPROVAL_VAR];
  }
});

async function workspaceRootActivities(handle: WorkflowHandle): Promise<number> {
  const { events } = await handle.fetchHistory();
  return (events ?? []).filter(
    (event) =>
      event.activityTaskScheduledEventAttributes?.activityType?.name === 'lifecycle-sandbox-client-workspace-root'
  ).length;
}

function rootMovingWorker(
  createWorker: ReturnType<typeof helpers>['createWorker'],
  backend: RootMovingSandboxClient,
  responses = [
    toolCallResponse('run_now', { cmd: 'first' }),
    toolCallResponse('run_approved', { cmd: 'second' }),
    textResponse('Done.'),
  ]
) {
  return createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider(responses),
        sandboxClientProviders: [new SandboxClientProvider('lifecycle', backend)],
      }),
    ],
  });
}

for (const resume of ['live', 'runState', 'sessionState'] as const) {
  test(`An approval resume (${resume}) reuses a sandbox whose backend moved its root`, async (t) => {
    const { createWorker, startWorkflow } = helpers(t);
    const backend = new RootMovingSandboxClient();
    const worker = await rootMovingWorker(createWorker, backend);
    const handle = await startWorkflow(sandboxRootResumeWorkflow, { args: [resume] });
    t.is(await worker.runUntil(handle.result()), '#1 ran first with KEY=\n#1 ran first,second with KEY=');
    t.is(await workspaceRootActivities(handle), 0, 'the Workflow knows the root from creating the sandbox');
    t.deepEqual(backend.liveSandboxes(), []);
  });
}

test('An approval resume in another Workflow asks the Worker once where the backend moved the root', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const backend = new RootMovingSandboxClient();
  const worker = await rootMovingWorker(createWorker, backend);
  const [paused, resumed] = await worker.runUntil(async () => {
    const paused = await startWorkflow(sandboxRootResumeWorkflow, { args: ['pause'] });
    const resumed = await startWorkflow(sandboxRootResumeWorkflow, { args: ['runState', await paused.result()] });
    t.is(await resumed.result(), '#1 ran first with KEY=\n#1 ran first,second with KEY=');
    return [paused, resumed];
  });
  t.is(await workspaceRootActivities(paused), 0);
  t.is(await workspaceRootActivities(resumed), 1);
  t.deepEqual(
    backend.events.filter((event) => /^(create|resume|delete)/.test(event)),
    ['create#1', 'resume#1', 'delete#1']
  );
});

for (const root of ['/workspace', '/custom']) {
  test(`A run in another Workflow resumes a saved sessionState with the manifest root at ${root}`, async (t) => {
    const { createWorker, startWorkflow } = helpers(t);
    const backend = new RootMovingSandboxClient();
    const worker = await rootMovingWorker(createWorker, backend, [
      toolCallResponse('run_now', { cmd: 'first' }),
      toolCallResponse('run_approved', { cmd: 'second' }),
      toolCallResponse('run_now', { cmd: 'second' }),
      textResponse('Done.'),
    ]);
    const resumed = await worker.runUntil(async () => {
      const saved = await (await startWorkflow(sandboxSavedSessionResumeWorkflow, { args: [root] })).result();
      const resumed = await startWorkflow(sandboxSavedSessionResumeWorkflow, { args: [root, saved] });
      t.is(await resumed.result(), '#1 ran first,second with KEY=');
      return resumed;
    });
    t.is(await workspaceRootActivities(resumed), 1);
    t.deepEqual(
      backend.events.filter((event) => /^(create|resume|delete)/.test(event)),
      ['create#1', 'resume#1', 'delete#1']
    );
  });
}

for (const continued of [false, true]) {
  const how = continued ? 'after continueAsNew' : 'in another Workflow';
  test(`A nested asTool() approval resumed ${how} reuses a sandbox whose backend moved its root`, async (t) => {
    const { createWorker, startWorkflow } = helpers(t);
    const backend = new RootMovingSandboxClient();
    const worker = await rootMovingWorker(createWorker, backend, [
      toolCallResponse('ask_sandbox', { input: 'run' }),
      toolCallResponse('run_now', { cmd: 'first' }),
      toolCallResponse('run_approved', { cmd: 'second' }),
      textResponse('nested done'),
      textResponse('Done.'),
    ]);
    const output = await worker.runUntil(async () => {
      if (continued) return (await startWorkflow(sandboxNestedRootResumeWorkflow, { args: [undefined, true] })).result();
      const paused = await startWorkflow(sandboxNestedRootResumeWorkflow, { args: [] });
      return (await startWorkflow(sandboxNestedRootResumeWorkflow, { args: [await paused.result()] })).result();
    });
    t.is(output, 'nested done');
    t.deepEqual(
      backend.events.filter((event) => /^(create|resume|delete)/.test(event)),
      ['create#1', 'resume#1', 'delete#1']
    );
  });
}

test("A SandboxAgent that names no model keeps its capabilities' providerData, such as compaction", async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);
  const settings: any[] = [];
  const provider = {
    getModel() {
      return {
        async getResponse(request: any) {
          settings.push(request.modelSettings);
          return textResponse('done');
        },
      } as any;
    },
  };
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: provider,
        sandboxClientProviders: [new SandboxClientProvider('fake', new FakeSandboxClient())],
      }),
    ],
  });
  await worker.runUntil(async () => t.is(await executeWorkflow(sandboxNoModelWorkflow), 'done'));
  t.is(settings[0]?.providerData?.context_management?.[0]?.type, 'compaction');
});

test('setArchiveLimits is forwarded to the persistWorkspace and hydrateWorkspace Activities', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const client = new FakeSandboxClient();
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([]),
        sandboxClientProviders: [new SandboxClientProvider('fake', client)],
      }),
    ],
  });

  await worker.runUntil(async () => {
    t.is(await executeWorkflow(sandboxArchiveLimitsWorkflow), 'ok');
  });

  t.deepEqual(client.session.archiveLimits, { maxInputBytes: 42 });
  t.deepEqual(client.session.hydrateCalls[0]!.archiveLimits, { maxInputBytes: 42 });
});

test('Sandbox configuration errors are raised inside the Workflow', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        // The lead's call in the last case, whose nested SandboxAgent fails before calling its model.
        modelProvider: new FakeModelProvider([toolCallResponse('ask_sandbox', { input: 'hello' })]),
        sandboxClientProviders: [new SandboxClientProvider('fake', new FakeSandboxClient())],
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(sandboxValidationWorkflow);
    t.is(result, 'OK');
  });
});

test('addTemporalSpans: sandbox exec emits sandbox:* spans carrying only safe metadata', async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  const agentSdk = installAgentSdkCollector();
  const worker = await createWorker({
    maxCachedWorkflows: 0,
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([]),
        sandboxClientProviders: [new SandboxClientProvider('fake', new FakeSandboxClient())],
        interceptorOptions: { addTemporalSpans: true, useOtelInstrumentation: false },
      }),
    ],
  });

  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
    interceptors: [new OpenAIAgentsTraceClientInterceptor({ addTemporalSpans: true })],
  });

  try {
    await worker.runUntil(async () => {
      await withTrace('sandbox-tracing-root', async () => {
        const handle = await wfClient.start(sandboxExecWorkflow, {
          taskQueue,
          workflowId: `sandbox-tracing-${Date.now()}`,
          workflowExecutionTimeout: '30 seconds',
        });
        t.is(await handle.result(), 'exit=0');
      });
    });

    const sandboxRecords = agentSdk.collector.records_().filter((r) => r.name.startsWith('sandbox:'));
    const named = (n: string) => sandboxRecords.filter((r) => r.name === n);

    t.is(named('sandbox:client-create').length, 0, 'client.create must not emit a workflow-side span');
    t.is(named('sandbox:client-create:result').length, 0, 'client.create must not emit a worker-side span');

    const wfExec = named('sandbox:session-exec');
    t.true(wfExec.length > 0, 'expected a workflow-side sandbox:session-exec span');
    t.true(wfExec.every((r) => typeof r.data?.sessionId === 'string'));

    const workerExec = named('sandbox:session-exec:result');
    t.true(workerExec.length > 0, 'expected a worker-side sandbox:session-exec:result span');
    t.true(workerExec.every((r) => typeof r.data?.sessionId === 'string'));
    t.true(
      workerExec.some((r) => r.data?.exitCode === 0),
      'worker exec span must record exitCode'
    );

    t.true(named('sandbox:session-read-file').length > 0, 'expected a workflow-side read span');
    const workerRead = named('sandbox:session-read-file:result');
    t.true(workerRead.length > 0, 'expected a worker-side read span');
    t.true(
      workerRead.some((r) => typeof r.data?.byteLength === 'number'),
      'worker read span must record byteLength'
    );

    t.true(named('sandbox:editor-create-file').length > 0, 'expected a workflow-side editor span');
    t.true(named('sandbox:editor-create-file:result').length > 0, 'expected a worker-side editor span');

    for (const r of sandboxRecords) {
      for (const key of Object.keys(r.data ?? {})) {
        t.true(PERMITTED_SPAN_DATA_KEYS.has(key), `sandbox span ${r.name} carries disallowed attribute '${key}'`);
      }
    }
  } finally {
    await agentSdk.teardown();
  }
});

for (const inGraph of [false, true]) {
  test(`A custom Handoff target ${inGraph ? 'also in the handoff graph' : 'returned only by the callback'} gets the sandbox config`, async (t) => {
    const { createWorker, executeWorkflow } = helpers(t);
    const session = new FakeSandboxSession();
    const client = new FakeSandboxClient(session);
    const worker = await createWorker({
      plugins: [
        new OpenAIAgentsPlugin({
          modelProvider: new FakeModelProvider([
            handoffResponse('transfer_to_desk'),
            toolCallResponse('ask_sandbox', { input: 'go' }),
            toolCallResponse('run_command', { cmd: 'echo hello' }),
            textResponse('sandbox done'),
            textResponse('vip answer'),
          ]),
          sandboxClientProviders: [new SandboxClientProvider('fake', client)],
        }),
      ],
    });

    await worker.runUntil(async () => {
      t.is(await executeWorkflow(customHandoffSandboxToolWorkflow, { args: [inGraph] }), 'vip: vip answer');
    });
    t.is(session.execCalls[0]?.cmd, 'echo hello');
  });
}

for (const [kind, event] of [
  ['agent-start', 'agent_start'],
  ['tool-start', 'agent_tool_start'],
] as const) {
  test(`A sandbox client that an ${event} listener replaces with a raw one never runs`, async (t) => {
    const { createWorker, startWorkflow } = helpers(t);
    const session = new FakeSandboxSession();
    const client = new FakeSandboxClient(session);
    const worker = await createWorker({
      plugins: [
        new OpenAIAgentsPlugin({
          modelProvider: new FakeModelProvider(
            kind === 'agent-start'
              ? [toolCallResponse('run_command', { cmd: 'echo hello' }), textResponse('Done.')]
              : [toolCallResponse('ask_sandbox', { input: 'go' })]
          ),
          sandboxClientProviders: [new SandboxClientProvider('fake', client)],
        }),
      ],
    });

    await worker.runUntil(async () => {
      const handle = await startWorkflow(sandboxClientChangedWorkflow, {
        args: [kind],
        workflowExecutionTimeout: '30 seconds',
      });
      const output = await handle.result();
      t.notRegex(output, /the raw sandbox client ran/);
      const { events } = await handle.fetchHistory();
      const activities = (events ?? []).map((e) => e.activityTaskScheduledEventAttributes?.activityType?.name);
      if (kind === 'agent-start') {
        // The run keeps the client it was checked with.
        t.is(output, 'Done.');
        t.true(
          activities.some((name) => name?.startsWith('fake')),
          activities.join(', ')
        );
        t.is(session.execCalls[0]?.cmd, 'echo hello');
      } else {
        // The nested run checks the sandbox config it uses as it starts; the failure is the tool's output.
        t.notRegex(output, /the nested SandboxAgent started/);
        t.regex(output, /^An error occurred while running the tool/);
        t.is(client.createCalls.length, 0);
      }
      await Worker.runReplayHistory(
        {
          workflowBundle: t.context.workflowBundle,
          plugins: [new OpenAIAgentsPlugin({ modelProvider: new FakeModelProvider([]) })],
        },
        await handle.fetchHistory()
      );
    });
  });
}

test('A custom Handoff returning a SandboxAgent without runConfig.sandbox fails the Workflow', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);
  const worker = await createWorker({
    plugins: [new OpenAIAgentsPlugin({ modelProvider: new FakeModelProvider([handoffResponse('transfer_to_desk')]) })],
  });

  await worker.runUntil(async () => {
    const err = await t.throwsAsync(executeWorkflow(customHandoffSandboxAgentWithoutConfigWorkflow));
    t.regex(String((err as any)?.cause?.message), /SandboxAgent execution requires `RunConfig.sandbox`/);
    t.is((err as any)?.cause?.type, 'AgentsWorkflowError');
  });
});

// The Unix local sandbox can't run on Windows, where CI also runs.
(process.platform === 'win32' ? test.skip : test)(
  'A UnixLocal sandbox resolves a Worker environment reference after create and after an approval resume',
  async (t) => {
    const { createWorker, executeWorkflow } = helpers(t);
    const variable = 'OPENAI_AGENTS_TEST_UNIX_LOCAL_SECRET';
    process.env[variable] = 'sk-unix-local-sentinel';
    const workspaceBaseDir = fs.mkdtempSync(path.join(os.tmpdir(), 'oa-env-ws-'));
    t.teardown(() => {
      delete process.env[variable];
      fs.rmSync(workspaceBaseDir, { recursive: true, force: true });
    });

    const worker = await createWorker({
      plugins: [
        new OpenAIAgentsPlugin({
          modelProvider: new FakeModelProvider([
            toolCallResponse('run_now', { cmd: 'echo "created=$API_KEY"' }),
            toolCallResponse('run_approved', { cmd: 'echo "resumed=$API_KEY"' }),
            textResponse('Done.'),
          ]),
          sandboxClientProviders: [
            new OnHostSandboxClientProvider('local', new UnixLocalSandboxClient({ workspaceBaseDir }), {
              workerName: 'unix-local',
            }),
          ],
          resolvableWorkerEnvVars: [variable],
        }),
      ],
    });

    await worker.runUntil(async () => {
      const output = await executeWorkflow(sandboxWorkerEnvWorkflow, { args: [variable] });
      t.regex(output, /created=sk-unix-local-sentinel/);
      t.regex(output, /resumed=sk-unix-local-sentinel/);
    });
  }
);

/** Waits until `query` on `handle` answers true. */
async function until(handle: { query(query: typeof sessionCreatedQuery): Promise<boolean> }): Promise<void> {
  while (!(await handle.query(sessionCreatedQuery))) await new Promise((resolve) => setTimeout(resolve, 100));
}

/** A Worker whose `lifecycle` provider is an on-host provider for `backend` named `workerName`. */
function onHostWorker(
  createWorker: ReturnType<typeof helpers>['createWorker'],
  backend: LifecycleSandboxClient,
  workerName: string,
  responses: ConstructorParameters<typeof FakeModelProvider>[0] = [],
  connection?: NativeConnection
): Promise<Worker> {
  return createWorker({
    ...(connection ? { connection } : {}),
    maxCachedWorkflows: 0,
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider(responses),
        sandboxClientProviders: [new OnHostSandboxClientProvider('lifecycle', backend, { workerName })],
      }),
    ],
  });
}

test.serial('Every operation on an on-host sandbox runs on the Worker that created it', async (t) => {
  const { createWorker, startWorkflow, taskQueue } = helpers(t);
  const first = new LifecycleSandboxClient();
  const second = new LifecycleSandboxClient();
  const commands = ['a', 'b', 'c', 'd', 'e', 'f'];

  const workerA = await onHostWorker(createWorker, first, 'A');
  await workerA.runUntil(async () => {
    // Only A runs, so A creates the sandbox.
    const handle = await startWorkflow(sandboxOperationsWorkflow, { args: [commands] });
    await until(handle);
    // B, as another process would, has its own connection. It also polls the shared queue, so it runs
    // some of the Workflow Tasks and could take any Activity sent there.
    const connection = await NativeConnection.connect({ address: t.context.env.address });
    t.teardown(() => connection.close());
    const workerB = await onHostWorker(createWorker, second, 'B', [], connection);
    await workerB.runUntil(async () => {
      await handle.signal(proceedSignal);
      const outputs = commands.map((_, i) => `#1 ran ${commands.slice(0, i + 1).join(',')} with KEY=`);
      t.is(await handle.result(), outputs.join('\n'));
    });
    const queues = scheduledQueues(await handle.fetchHistory());
    t.deepEqual(queues[0], ['lifecycle-sandbox-client-create', taskQueue]);
    for (const [type, queue] of queues.slice(1)) t.is(queue, `${taskQueue}/lifecycle@A`, type);
  });

  t.deepEqual(first.events, ['create#1', ...commands.map(() => 'exec#1'), 'delete#1']);
  t.deepEqual(second.events, []);
});

test.serial('After its Worker restarts, an on-host sandbox is lost and deleting it does nothing', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const before = new LifecycleSandboxClient();
  const after = new LifecycleSandboxClient();

  const handle = await (
    await onHostWorker(createWorker, before, 'restarted')
  ).runUntil(async () => {
    const handle = await startWorkflow(sandboxOperationsWorkflow, { args: [['a']] });
    await until(handle);
    return handle;
  });

  // The new process has the same name, so it polls the same queue, but holds no sandbox.
  const output = await (
    await onHostWorker(createWorker, after, 'restarted')
  ).runUntil(async () => {
    await handle.signal(proceedSignal);
    return handle.result();
  });

  t.is(output, 'failed: SandboxLost');
  t.deepEqual(after.events, []);
  // The plugin doesn't clean up the sandbox the old process left behind.
  t.deepEqual(before.liveSandboxes(), [1]);
  const { events } = await handle.fetchHistory();
  const failures = (events ?? []).filter((event) => event.activityTaskFailedEventAttributes);
  t.is(failures.length, 1, 'SandboxLost is not retried, and the delete succeeds');
  t.true(failures[0]!.activityTaskFailedEventAttributes!.failure!.applicationFailureInfo!.nonRetryable!);
});

test.serial('An on-host sandbox keeps its Task Queue through an approval pause and continue-as-new', async (t) => {
  const { createWorker, startWorkflow, taskQueue } = helpers(t);
  const backend = new LifecycleSandboxClient();
  const worker = await onHostWorker(createWorker, backend, 'continued', [
    toolCallResponse('run_now', { cmd: 'first' }),
    toolCallResponse('run_approved', { cmd: 'second' }),
    textResponse('Done.'),
  ]);

  await worker.runUntil(async () => {
    const handle = await startWorkflow(sandboxContinueAsNewWorkflow);
    t.is(await handle.result(), '#1 ran first with KEY=\n#1 ran first,second with KEY=');
    const continued = await handle.client.getHandle(handle.workflowId).fetchHistory();
    // The workspace root lookup names no session, so like create it runs on the Workflow's Task Queue.
    const queues = scheduledQueues(continued).filter(
      ([type]) => type.startsWith('lifecycle-') && type !== 'lifecycle-sandbox-client-workspace-root'
    );
    t.deepEqual(queues[0], ['lifecycle-sandbox-client-resume', `${taskQueue}/lifecycle@continued`]);
    for (const [type, queue] of queues) t.is(queue, `${taskQueue}/lifecycle@continued`, type);
    t.true(scheduledQueues(continued).some(([type, queue]) => type === 'invokeModelActivity' && queue === taskQueue));
  });

  // The new run resumed the sandbox the first one created, which its Worker still held.
  t.deepEqual(
    backend.events.filter((event) => /^(create|resume|delete)/.test(event)),
    ['create#1', 'delete#1']
  );
});

test.serial('Two Workers that share a plugin on their NativeConnection each serve their own on-host sandboxes', async (t) => {
  const backend = new LifecycleSandboxClient();
  const plugin = new OpenAIAgentsPlugin({
    modelProvider: new FakeModelProvider([]),
    sandboxClientProviders: [new OnHostSandboxClientProvider('lifecycle', backend, { workerName: 'shared' })],
  });
  const connection = await NativeConnection.connect({ address: t.context.env.address, plugins: [plugin] });
  t.teardown(() => connection.close());
  const queues = [`${helpers(t).taskQueue}-first`, `${helpers(t).taskQueue}-second`];
  const workers = await Promise.all(
    queues.map((taskQueue) =>
      Worker.create({
        connection,
        namespace: t.context.env.namespace,
        workflowBundle: t.context.workflowBundle,
        taskQueue,
        maxCachedWorkflows: 0,
      })
    )
  );

  const run = async (taskQueue: string) => {
    const handle = await t.context.env.client.workflow.start(sandboxOperationsWorkflow, {
      args: [['a']],
      taskQueue,
      workflowId: `${taskQueue}-workflow`,
    });
    await until(handle);
    await handle.signal(proceedSignal);
    const output = await handle.result();
    return { output, queues: scheduledQueues(await handle.fetchHistory()) };
  };
  const [first, second] = await workers[0]!.runUntil(workers[1]!.runUntil(Promise.all(queues.map(run))));

  t.deepEqual([first!.output, second!.output].sort(), ['#1 ran a with KEY=', '#2 ran a with KEY=']);
  for (const [{ queues: scheduled }, taskQueue] of [
    [first!, queues[0]!],
    [second!, queues[1]!],
  ] as const) {
    t.deepEqual(scheduled[0], ['lifecycle-sandbox-client-create', taskQueue]);
    for (const [type, queue] of scheduled.slice(1)) t.is(queue, `${taskQueue}/lifecycle@shared`, type);
  }
});

/** Starts `worker` and shuts it down as soon as it runs. */
async function runBriefly(worker: Worker): Promise<void> {
  // The main Worker runs once its extra Workers exist.
  await worker.runUntil(async () => {
    while (worker.getState() !== 'RUNNING') await new Promise((resolve) => setTimeout(resolve, 10));
  });
}

/** Runs `worker` briefly and returns the extra Workers it started. */
async function extraWorkersOf(worker: Worker): Promise<Worker[]> {
  const created: Worker[] = [];
  const create = Worker.create.bind(Worker);
  const spy = async (options: WorkerOptions) => {
    const extra = await create(options);
    created.push(extra);
    return extra;
  };
  Object.assign(Worker, { create: spy });
  try {
    await runBriefly(worker);
  } finally {
    Object.assign(Worker, { create });
  }
  return created;
}

test.serial("An on-host provider's extra Worker takes its main Worker's Activity limits and grace time", async (t) => {
  const { createWorker } = helpers(t);
  const plugins = [
    new OpenAIAgentsPlugin({
      modelProvider: new FakeModelProvider([]),
      sandboxClientProviders: [new OnHostSandboxClientProvider('lifecycle', new LifecycleSandboxClient())],
    }),
  ];
  const limited = await createWorker({
    plugins,
    maxConcurrentActivityTaskExecutions: 3,
    maxActivitiesPerSecond: 7,
    maxTaskQueueActivitiesPerSecond: 11,
    shutdownGraceTime: '2s',
  });
  const [extra] = await extraWorkersOf(limited);
  t.deepEqual(extra!.options.tuner.activityTaskSlotSupplier, { type: 'fixed-size', numSlots: 3 });
  t.is(extra!.options.maxActivitiesPerSecond, 7);
  t.is(extra!.options.maxTaskQueueActivitiesPerSecond, 11);
  t.is(extra!.options.shutdownGraceTimeMs, 2000);

  const tuned = await createWorker({
    plugins,
    tuner: {
      tunerOptions: { targetMemoryUsage: 0.8, targetCpuUsage: 0.9 },
      activityTaskSlotOptions: { minimumSlots: 2, maximumSlots: 6, rampThrottle: '20ms' },
    },
  });
  const [tunedExtra] = await extraWorkersOf(tuned);
  t.deepEqual(tunedExtra!.options.tuner.activityTaskSlotSupplier, tuned.options.tuner.activityTaskSlotSupplier);
  t.like(tunedExtra!.options.tuner.activityTaskSlotSupplier, { type: 'resource-based', maximumSlots: 6 });
});

test.serial('With an on-host provider, a Worker refuses another plugin on its NativeConnection', async (t) => {
  const { createWorker } = helpers(t);
  const other = new SimplePlugin({ name: 'other-plugin' });
  const connection = await NativeConnection.connect({ address: t.context.env.address, plugins: [other] });
  t.teardown(() => connection.close());
  const plugin = (sandboxClientProviders: Array<SandboxClientProvider | OnHostSandboxClientProvider>) =>
    new OpenAIAgentsPlugin({ modelProvider: new FakeModelProvider([]), sandboxClientProviders });

  const err = await t.throwsAsync(
    createWorker({
      connection,
      plugins: [plugin([new OnHostSandboxClientProvider('lifecycle', new LifecycleSandboxClient())])],
    }),
    { instanceOf: TypeError }
  );
  t.regex(err!.message, /Plugin 'other-plugin' is on the NativeConnection/);
  t.regex(err!.message, /Register it on the Worker \(Worker\.create plugins\)/);
  // Without an on-host provider, or with the other plugin on the Worker, it starts.
  await runBriefly(await createWorker({ connection, plugins: [plugin([])] }));
  await runBriefly(
    await createWorker({
      plugins: [other, plugin([new OnHostSandboxClientProvider('lifecycle', new LifecycleSandboxClient())])],
    })
  );
});
