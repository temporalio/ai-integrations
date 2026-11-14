import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { setTracingDisabled, withTrace } from '@openai/agents-core';
import { UnixLocalSandboxClient } from '@openai/agents-core/sandbox/local';
import { WorkflowClient } from '@temporalio/client';
import { Worker } from '@temporalio/worker';
import { helpers } from '@temporalio/test-helpers';
import { OpenAIAgentsPlugin, OpenAIAgentsTraceClientInterceptor, SandboxClientProvider } from '..';
import { installAgentSdkCollector } from './helpers/openai-agents-spans';
import { makeTestFunction } from './helpers/test-fn';
import { FakeModelProvider, handoffResponse, textResponse, toolCallResponse } from './stubs/openai-agents';
import { FakeSandboxClient, FakeSandboxSession, LifecycleSandboxClient } from './stubs/sandbox-fakes';
import {
  approvalPausedQuery,
  approvalResumeSignal,
  customHandoffSandboxAgentWithoutConfigWorkflow,
  customHandoffSandboxToolWorkflow,
  sandboxAgentWorkflow,
  sandboxApprovalMountWorkflow,
  sandboxApprovalSequenceWorkflow,
  sandboxApprovalResumeWorkflow,
  sandboxArchiveLimitsWorkflow,
  sandboxClientChangedWorkflow,
  sandboxExecWorkflow,
  sandboxManifestResumeWorkflow,
  sandboxNestedApprovalWorkflow,
  sandboxNoModelWorkflow,
  sandboxResumeEnvironmentWorkflow,
  sandboxValidationWorkflow,
  sandboxWorkerEnvWorkflow,
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
const OTHER_APPROVAL_VAR = 'OPENAI_AGENTS_TEST_OTHER_APPROVAL_VALUE';

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
        modelParams: { startToCloseTimeout: '30s' },
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
        modelParams: { startToCloseTimeout: '30s' },
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
        modelParams: { startToCloseTimeout: '30s' },
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

test('A nested approval behind a copy of an asTool() tool in a sandbox run with an opaque mount fails the resume', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([
          toolCallResponse('ask_nested', { input: 'q' }),
          toolCallResponse('confirmAction', { reason: 'approved' }),
        ]),
        sandboxClientProviders: [new SandboxClientProvider('fake', new FakeSandboxClient())],
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(sandboxNestedApprovalWorkflow, {
      args: [true, 'sandbox-parent', false, 'opaque'],
    });
    t.regex(result, /^AgentsWorkflowError: The approval for 'confirmAction' of agent 'Nested' can't be resumed/);
    t.regex(result, /executions=0$/);
  });
});

for (const mount of ['none', 'resumable', 'in-container', 'opaque'] as const) {
  for (const withReference of [false, true]) {
    const replaced = mount === 'opaque';
    // Serial: a Worker installs its allowlist process-wide, so a Worker created alongside would switch it off.
    test.serial(
      `An approval resume of a sandbox with ${mount === 'none' ? 'no' : mount === 'resumable' ? 'a resumable' : `an ${mount}`} mount${withReference ? ' and a workerEnvValue' : ''} ${replaced ? 'replaces' : 'keeps'} it and leaks none`,
      async (t) => {
        const { createWorker, executeWorkflow } = helpers(t);
        const backend = new LifecycleSandboxClient();
        const worker = await createWorker({
          plugins: [
            new OpenAIAgentsPlugin({
              modelProvider: new FakeModelProvider([
                toolCallResponse('run_now', { cmd: 'first' }),
                toolCallResponse('run_approved', { cmd: 'second' }),
                textResponse('Done.'),
              ]),
              sandboxClientProviders: [new SandboxClientProvider('lifecycle', backend)],
              resolvableWorkerEnvVars: [APPROVAL_VAR],
            }),
          ],
        });

        process.env[APPROVAL_VAR] = 'resolved-on-worker';
        try {
          const output = await worker.runUntil(
            executeWorkflow(sandboxApprovalMountWorkflow, { args: [mount, withReference ? APPROVAL_VAR : undefined] })
          );
          const key = `KEY=${withReference ? 'resolved-on-worker' : ''}`;
          if (replaced) {
            // As upstream, a sandbox with a mount upstream can't verify is replaced.
            t.is(output, `#1 ran first with ${key}\n#2 ran second with ${key}`);
            t.deepEqual(
              backend.events.filter((event) => /^(create|resume|delete)/.test(event)),
              ['create#1', 'delete#1', 'create#2', 'delete#2']
            );
          } else {
            t.is(output, `#1 ran first with ${key}\n#1 ran first,second with ${key}`);
            t.deepEqual(
              backend.events.filter((event) => /^(create|resume|delete)/.test(event)),
              ['create#1', 'delete#1']
            );
          }
          t.deepEqual(backend.liveSandboxes(), []);
        } finally {
          delete process.env[APPROVAL_VAR];
        }
      }
    );
  }
}

for (const mount of ['resumable', 'in-container'] as const) {
  // Serial: a Worker installs its allowlist process-wide, so a Worker created alongside would switch it off.
  test.serial(
    `A second approval resume, with a new client and another Worker variable, ${mount === 'resumable' ? 'resumes the sandbox with the new variable' : 'replaces a sandbox with an in-container mount'}`,
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
            sandboxClientProviders: [new SandboxClientProvider('lifecycle', backend)],
            resolvableWorkerEnvVars: [APPROVAL_VAR, OTHER_APPROVAL_VAR],
          }),
        ],
      });

      process.env[APPROVAL_VAR] = 'resolved-on-worker';
      process.env[OTHER_APPROVAL_VAR] = 'other-on-worker';
      try {
        const output = await worker.runUntil(
          executeWorkflow(sandboxApprovalSequenceWorkflow, {
            args: [mount, [APPROVAL_VAR, APPROVAL_VAR, OTHER_APPROVAL_VAR]],
          })
        );
        const events = backend.events.filter((event) => /^(create|resume|delete)/.test(event));
        if (mount === 'resumable') {
          t.is(output, '#1 ran first,second,third with KEY=other-on-worker');
          t.deepEqual(events, ['create#1', 'resume#1', 'delete#1']);
        } else {
          t.is(output, '#2 ran third with KEY=other-on-worker');
          t.deepEqual(events, ['create#1', 'delete#1', 'create#2', 'delete#2']);
        }
        t.deepEqual(backend.liveSandboxes(), []);
      } finally {
        delete process.env[APPROVAL_VAR];
        delete process.env[OTHER_APPROVAL_VAR];
      }
    }
  );
}

test.serial(
  'An approval resume of a sandbox with an opaque mount deletes it through the client when its Worker lost it',
  async (t) => {
    const { createWorker, startWorkflow } = helpers(t);
    const backend = new LifecycleSandboxClient();
    const workerWith = (responses: ConstructorParameters<typeof FakeModelProvider>[0]) =>
      createWorker({
        maxCachedWorkflows: 0,
        plugins: [
          new OpenAIAgentsPlugin({
            modelProvider: new FakeModelProvider(responses),
            sandboxClientProviders: [new SandboxClientProvider('lifecycle', backend)],
          }),
        ],
      });

    const first = await workerWith([
      toolCallResponse('run_now', { cmd: 'first' }),
      toolCallResponse('run_approved', { cmd: 'second' }),
    ]);
    const handle = await first.runUntil(async () => {
      const handle = await startWorkflow(sandboxApprovalMountWorkflow, { args: ['opaque', undefined, true] });
      while (!(await handle.query(approvalPausedQuery))) await new Promise((resolve) => setTimeout(resolve, 100));
      return handle;
    });

    // A new Worker caches no sessions, as after a restart.
    const second = await workerWith([textResponse('Done.')]);
    const output = await second.runUntil(async () => {
      await handle.signal(approvalResumeSignal);
      return await handle.result();
    });

    t.is(output, '#1 ran first with KEY=\n#2 ran second with KEY=');
    t.deepEqual(
      backend.events.filter((event) => /^(create|resume|delete|client-delete)/.test(event)),
      ['create#1', 'client-delete#1', 'create#2', 'delete#2']
    );
    t.deepEqual(backend.liveSandboxes(), []);
  }
);

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
          sandboxClientProviders: [new SandboxClientProvider('local', new UnixLocalSandboxClient({ workspaceBaseDir }))],
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
