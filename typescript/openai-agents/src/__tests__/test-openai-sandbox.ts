import { setTracingDisabled, withTrace } from '@openai/agents-core';
import { WorkflowClient } from '@temporalio/client';
import { Worker } from '@temporalio/worker';
import { helpers } from '@temporalio/test-helpers';
import { OpenAIAgentsPlugin, OpenAIAgentsTraceClientInterceptor, SandboxClientProvider } from '..';
import { installAgentSdkCollector } from './helpers/openai-agents-spans';
import { makeTestFunction } from './helpers/test-fn';
import { FakeModelProvider, handoffResponse, textResponse, toolCallResponse } from './stubs/openai-agents';
import { FakeSandboxClient, FakeSandboxSession } from './stubs/sandbox-fakes';
import {
  customHandoffSandboxAgentWithoutConfigWorkflow,
  customHandoffSandboxToolWorkflow,
  sandboxAgentWorkflow,
  sandboxApprovalResumeWorkflow,
  sandboxArchiveLimitsWorkflow,
  sandboxClientChangedWorkflow,
  sandboxExecWorkflow,
  sandboxManifestResumeWorkflow,
  sandboxValidationWorkflow,
} from './workflows/openai-sandbox';

setTracingDisabled(false);

const test = makeTestFunction({
  workflowsPath: require.resolve('./workflows/openai-sandbox'),
  plugins: [new OpenAIAgentsPlugin({ modelProvider: new FakeModelProvider([]) })],
});

const PERMITTED_SPAN_DATA_KEYS = new Set(['sessionId', 'port', 'exitCode', 'byteLength', 'length', 'count']);

const MANIFEST_SECRET_VAR = 'OPENAI_AGENTS_TEST_MANIFEST_SECRET';
const MANIFEST_SECRET = 'sk-approval-resume-sentinel';

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
  const { createWorker, executeWorkflow } = helpers(t);

  const client = new FakeSandboxClient();
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([
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
      t.is(await executeWorkflow(sandboxApprovalResumeWorkflow), 'Done.');
    });
  } finally {
    delete process.env[MANIFEST_SECRET_VAR];
  }

  t.true(client.resumeCalls >= 1, 'the preserved session is re-established through resume()');
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
        modelProvider: new FakeModelProvider([]),
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
        t.regex(output, /An error occurred while running the tool.*runOptions\.sandbox\.client must be created via/);
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
