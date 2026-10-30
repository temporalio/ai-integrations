// Integration tests that need a real Worker. Pure-function tests live in
// the sibling `*.test.ts` files.
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { setTracingDisabled } from '@openai/agents-core';
import type { MCPServer, Model, ModelProvider, ModelRequest, ModelResponse, StreamEvent } from '@openai/agents-core';
import type { ExecutionContext } from 'ava';
import { WorkflowClient, WorkflowFailedError, type WorkflowHandle } from '@temporalio/client';
import { CancelledFailure } from '@temporalio/common';
import { temporal } from '@temporalio/proto';
import { helpers, type BaseContext } from '@temporalio/test-helpers';
import { Worker, bundleWorkflowCode } from '@temporalio/worker';
import { WorkflowStreamClient } from '@temporalio/workflow-streams/client';
import {
  OpenAIAgentsPlugin,
  OpenAIAgentsTraceClientInterceptor,
  StatefulMCPServerProvider,
  StatelessMCPServerProvider,
} from '..';
import {
  agentHooksWorkflow,
  agentToolRejectedWorkflow,
  agentToolRunConfigWorkflow,
  agentToolStreamWorkflow,
  agentToolOptionsReuseWorkflow,
  customHandoffWorkflow,
  runModelOverrideWorkflow,
  sharedAgentApprovalWorkflow,
  summaryAgentWorkflow,
  localActivityAgentWorkflow,
  mcpFactoryArgWorkflow,
  nestedToolCancelWorkflow,
  runningAgentRejectedWorkflow,
  summaryOverrideStringWorkflow,
  statefulMcpNotConnectedWorkflow,
  statefulMcpIsolationWorkflow,
  statefulMcpHeartbeatTimeoutWorkflow,
  statefulMcpSlowConnectHeartbeatWorkflow,
  streamingAgentWorkflow,
  streamingLocalActivityWorkflow,
  streamingNoTopicWorkflow,
  type AgentToolRejection,
  type RunningAgentRejection,
} from './workflows/openai-agents';
import type { AgentCopyCase } from './workflows/openai-agents-copies';
import { makeTestFunction } from './helpers/test-fn';
import {
  FakeModelProvider,
  multiToolCallResponse,
  textResponse,
  toolCallResponse,
  type FakeModel,
} from './stubs/openai-agents';
import { StreamingFakeModelProvider, handoffResponse, streamingTextEvents } from './stubs/openai-agents-fakes';
import EventType = temporal.api.enums.v1.EventType;

// Upstream auto-disables agent-SDK tracing under NODE_ENV=test; opt back in for the integration tests.
setTracingDisabled(false);

// Compile-time check: plugin + client-interceptor modelParams reject function/object summary.
{
  const _provider = null! as FakeModelProvider;
  // @ts-expect-error — modelParams.summary must be string, not function/object
  new OpenAIAgentsPlugin({ modelProvider: _provider, modelParams: { summary: { provide: () => 'x' } } });
  // @ts-expect-error — modelParams.summary must be string, not function/object
  new OpenAIAgentsTraceClientInterceptor({ modelParams: { summary: { provide: () => 'x' } } });
}

const test = makeTestFunction({
  workflowsPath: require.resolve('./workflows/openai-agents'),
  workflowInterceptorModules: [require.resolve('@temporalio/openai-agents/workflow-interceptor')],
  // The plugin's configureBundler hook prepends the polyfill installer to the
  // webpack entry array. The model provider stub is never invoked at bundle time.
  plugins: [new OpenAIAgentsPlugin({ modelProvider: new FakeModelProvider([]) })],
});

/** Replays a Workflow's history with a Worker whose model provider must not be called. */
async function replay(t: ExecutionContext<BaseContext>, handle: WorkflowHandle): Promise<void> {
  await Worker.runReplayHistory(
    {
      workflowBundle: t.context.workflowBundle,
      plugins: [new OpenAIAgentsPlugin({ modelProvider: new FakeModelProvider([]) })],
    },
    await handle.fetchHistory()
  );
}

test('Local activity mode uses local activities for model calls', async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Local activity response')]),
        modelParams: { useLocalActivity: true, startToCloseTimeout: '60s' },
      }),
    ],
  });

  // Interceptor on the start-side wfClient is what injects the config header.
  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
    interceptors: [
      new OpenAIAgentsTraceClientInterceptor({
        modelParams: { useLocalActivity: true, startToCloseTimeout: '60s' },
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await wfClient.start(localActivityAgentWorkflow, {
      taskQueue,
      workflowId: `local-activity-${Date.now()}`,
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.is(result, 'Local activity response');

    const { events } = await handle.fetchHistory();

    // Local activities show up as MarkerRecorded ("core_local_activity"), not ActivityTaskScheduled.
    const markerEvents = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_MARKER_RECORDED) ?? [];
    t.true(
      markerEvents.length > 0,
      `Expected MarkerRecorded events for local activities in history, got ${markerEvents.length}`
    );

    const activityScheduledEvents =
      events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED) ?? [];
    const modelActivities = activityScheduledEvents.filter(
      (e) => e.activityTaskScheduledEventAttributes?.activityType?.name === 'invokeModelActivity'
    );
    t.is(modelActivities.length, 0, `Expected no regular invokeModelActivity, got ${modelActivities.length}`);
  });
});

test('A Workflow module with ES module imports loads with one copy of @openai/agents-core', async (t) => {
  const { createWorker, taskQueue } = helpers(t);
  const plugin = new OpenAIAgentsPlugin({
    modelProvider: new FakeModelProvider([textResponse('ESM module response')]),
  });
  // Bundle the ES module source, as the Worker does for a `.ts` Workflow file run from source.
  const workflowBundle = await bundleWorkflowCode({
    workflowsPath: require.resolve('../../src/__tests__/workflows/openai-agents-esm.mjs'),
    plugins: [plugin],
  });
  const esModuleBuild = new Set(workflowBundle.code.match(/@openai[\\/]agents-core[\\/]dist[\\/][^"'\s]*\.mjs/g));
  t.deepEqual([...esModuleBuild].slice(0, 3), [], `${esModuleBuild.size} modules from the ES module build`);

  const worker = await createWorker({ workflowBundle, plugins: [plugin] });
  const result = await worker.runUntil(
    t.context.env.client.workflow.execute('esmModuleAgentWorkflow', {
      taskQueue,
      workflowId: `esm-module-${Date.now()}`,
      args: ['Hello'],
    })
  );
  t.is(result, 'ESM module response');
});

test('A Workflow module with ES module imports bundles one copy of @openai/agents-core with async chunks off', async (t) => {
  const plugin = new OpenAIAgentsPlugin({ modelProvider: new FakeModelProvider([]) });
  // These configs avoid the async chunk, so a bundle with both builds of @openai/agents-core would
  // still load, and the plugin's instanceof checks would miss objects from one of them.
  const setups: Record<string, NonNullable<Parameters<typeof bundleWorkflowCode>[0]['webpackConfigHook']>> = {
    'output.asyncChunks: false': (config) => ({ ...config, output: { ...config.output, asyncChunks: false } }),
    "dynamicImportMode: 'eager'": (config) => ({
      ...config,
      module: { ...config.module, parser: { ...config.module?.parser, javascript: { dynamicImportMode: 'eager' } } },
    }),
  };
  for (const [name, webpackConfigHook] of Object.entries(setups)) {
    const { code } = await bundleWorkflowCode({
      workflowsPath: require.resolve('../../src/__tests__/workflows/openai-agents-esm.mjs'),
      plugins: [plugin],
      webpackConfigHook,
    });
    const esModuleBuild = new Set(code.match(/@openai[\\/]agents-core[\\/]dist[\\/][^"'\s]*\.mjs/g));
    t.deepEqual([...esModuleBuild].slice(0, 3), [], `${name}: ${esModuleBuild.size} modules from the ES module build`);
  }
});

function* mcpFactoryArgGenerator() {
  yield toolCallResponse('get_time', {});
  yield textResponse('The time for tenant-42 is 2026-01-01.');
}

test('factoryArgument is passed through to MCP activities', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  let receivedFactoryArg: unknown;
  const testMcp = new StatelessMCPServerProvider('testMcp', (factoryArgument?: unknown): MCPServer => {
    receivedFactoryArg = factoryArgument;
    return {
      cacheToolsList: false,
      name: 'testMcp',
      async connect() {},
      async close() {},
      async listTools() {
        return [
          {
            name: 'get_time',
            description: 'Returns current time',
            inputSchema: {
              type: 'object' as const,
              properties: {},
              required: [] as string[],
              additionalProperties: false,
            },
          },
        ];
      },
      async callTool(_toolName: string, _args: Record<string, unknown> | null) {
        t.deepEqual(factoryArgument, { tenantId: 'tenant-42' }, 'factoryArgument should be passed to callTool');
        return [{ type: 'text', text: '2026-01-01' }];
      },
      async invalidateToolsCache() {},
    };
  });
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider(() => mcpFactoryArgGenerator()),
        mcpServerProviders: [testMcp],
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(mcpFactoryArgWorkflow, {
      args: ['What time is it?'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.truthy(result, 'Workflow should complete successfully');

    t.deepEqual(receivedFactoryArg, { tenantId: 'tenant-42' }, 'factoryArgument should be passed to listTools');
  });
});

test('summary string is passed through to model activity', async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('Summary test response')]),
        modelParams: { summary: 'Custom model summary' },
      }),
    ],
  });

  // Interceptor on the start-side wfClient is what injects the config header.
  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
    interceptors: [
      new OpenAIAgentsTraceClientInterceptor({
        modelParams: { summary: 'Custom model summary' },
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await wfClient.start(summaryOverrideStringWorkflow, {
      taskQueue,
      workflowId: `summary-override-${Date.now()}`,
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.is(result, 'Summary test response');

    const { events } = await handle.fetchHistory();
    const activityScheduledEvents =
      events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED) ?? [];
    t.true(
      activityScheduledEvents.length >= 1,
      `Expected at least 1 activity scheduled event, got ${activityScheduledEvents.length}`
    );

    const modelEvent = activityScheduledEvents.find(
      (e) => e?.activityTaskScheduledEventAttributes?.activityType?.name === 'invokeModelActivity'
    );
    t.truthy(modelEvent, 'Expected invokeModelActivity in history');
    const userMetadata = (modelEvent as any)?.userMetadata;
    t.truthy(userMetadata, 'Expected userMetadata on activity scheduled event');
    if (userMetadata) {
      const summaryPayload = userMetadata?.summary;
      t.truthy(summaryPayload, 'Expected summary payload in userMetadata');
      if (summaryPayload) {
        const summaryText = Buffer.from(summaryPayload.data).toString('utf-8');
        t.true(
          summaryText.includes('Custom model summary'),
          `Expected summary metadata to contain 'Custom model summary', got: ${summaryText}`
        );
      }
    }
  });
});

test('agent.asTool() runs take the run model, model settings and runner defaults', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const requests: { model: string; instructions: string; modelSettings: Record<string, unknown> }[] = [];
  let leadCalls = 0;
  const provider = {
    getModel(name?: string) {
      return {
        async getResponse(request: any) {
          requests.push({
            model: name ?? '',
            instructions: request.systemInstructions,
            modelSettings: request.modelSettings,
          });
          if (request.systemInstructions === 'Answer as the expert.') return textResponse('expert answer');
          if (request.systemInstructions === 'Answer as the specialist.') return textResponse('specialist answer');
          return leadCalls++ === 0
            ? multiToolCallResponse([
                { name: 'ask_expert', args: { input: 'q1' } },
                { name: 'ask_specialist', args: { input: 'q2' } },
              ])
            : textResponse('done');
        },
      } as unknown as FakeModel;
    },
  };
  const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })] });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(agentToolRunConfigWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(await handle.result(), 'done');

    const first = (instructions: string) => requests.find((r) => r.instructions === instructions);
    // A nested agent with no model gets the run model; one with its own model keeps it, as natively.
    t.is(first('Answer as the expert.')?.model, 'run-model');
    t.is(first('Answer as the specialist.')?.model, 'own-model');
    for (const instructions of ['Use the tools.', 'Answer as the expert.', 'Answer as the specialist.']) {
      t.is(first(instructions)?.modelSettings.temperature, 0.33, instructions);
    }
    // toolChoice picks the outer agent's tools, so nested runs don't take it.
    t.is(first('Use the tools.')?.modelSettings.toolChoice, 'auto');
    t.is(first('Answer as the expert.')?.modelSettings.toolChoice, undefined);

    const { events } = await handle.fetchHistory();
    const modelActivities = (events ?? []).filter(
      (e) => e.activityTaskScheduledEventAttributes?.activityType?.name === 'invokeModelActivity'
    );
    t.is(modelActivities.length, 4);
    for (const e of modelActivities) {
      t.is(Number(e.activityTaskScheduledEventAttributes?.startToCloseTimeout?.seconds), 17);
      const summary = Buffer.from(e.userMetadata?.summary?.data ?? []).toString('utf-8');
      t.true(summary.includes('runner default'), summary);
    }
    await replay(t, handle);
  });
});

test('agent.asTool() setups that would run a model or sandbox in the Workflow fail before running', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const worker = await createWorker();
  const expected: Record<AgentToolRejection, [string, RegExp]> = {
    'model-object': ['AgentsWorkflowError', /Agent 'Inline' has a Model object/],
    'nested-handoff-model-object': ['AgentsWorkflowError', /Agent 'Inline' has a Model object/],
    'model-provider': ['AgentsWorkflowError', /asTool\('ask_nested'\) sets runConfig.modelProvider/],
    'raw-sandbox': ['SandboxConfigurationError', /asTool\('ask_nested'\) runConfig.sandbox.client must be created/],
    'raw-sandbox-run-options': [
      'SandboxConfigurationError',
      /asTool\('ask_nested'\) runOptions.sandbox.client must be created/,
    ],
    unrecorded: ['AgentsWorkflowError', /tool 'ask_nested' looks like an agent.asTool\(\) tool/],
  };

  await worker.runUntil(async () => {
    for (const [kind, [type, message]] of Object.entries(expected)) {
      const handle = await startWorkflow(agentToolRejectedWorkflow, {
        args: [kind as AgentToolRejection],
        workflowExecutionTimeout: '30 seconds',
      });
      const result = await handle.result();
      t.true(result.startsWith(`${type}: `), `${kind}: ${result}`);
      t.regex(result, message, kind);
      const { events } = await handle.fetchHistory();
      t.false(
        (events ?? []).some((e) => e.activityTaskScheduledEventAttributes != null),
        `${kind}: no Activity is scheduled`
      );
    }
  });
});

test('Agents picked or changed during the run are checked before they call their model', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const models: string[] = [];
  const provider = {
    getModel(model: string) {
      models.push(model);
      return {
        async getResponse(request: any) {
          if (request.systemInstructions === 'Use the tool.') return toolCallResponse('ask_nested', { input: 'q' });
          if (request.systemInstructions === 'Answer as nested.') return textResponse('nested answer');
          return handoffResponse('transfer_to_desk');
        },
      } as unknown as FakeModel;
    },
  };
  const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })] });
  // [expected result, model Activities the run schedules]
  const expected: Record<RunningAgentRejection, [RegExp, number]> = {
    'handoff-callback-model-object': [/^AgentsWorkflowError: .*Agent 'Picked' has a Model object/, 1],
    'handoff-callback-model-provider': [
      /^AgentsWorkflowError: .*asTool\('ask_nested'\) sets runConfig.modelProvider/,
      1,
    ],
    'agent-start-model-object': [/^AgentsWorkflowError: .*Agent 'Lead' has a Model object/, 0],
    // A nested run's failure is the tool's output, as upstream; the run ends on it.
    'nested-agent-start-model-object': [
      /^An error occurred while running the tool.*Agent 'Nested' has a Model object/,
      1,
    ],
    // The nested Runner's run checks the config it uses as it starts, inside the tool, so upstream
    // returns the failure as the tool's output.
    'tool-start-model-provider': [
      /^An error occurred while running the tool.*agent 'Nested' has a model provider that would call models/,
      1,
    ],
    'tool-start-run-model-object': [
      /^An error occurred while running the tool.*agent 'Nested' has a Model object as runConfig.model/,
      1,
    ],
    'input-builder-model-provider': [
      /^An error occurred while running the tool.*agent 'Nested' has a model provider that would call models/,
      1,
    ],
    'tool-start-clear-model-provider': [
      /^An error occurred while running the tool.*agent 'Nested' has a model provider that would call models/,
      1,
    ],
    'tool-option-model-provider-undefined': [
      /^AgentsWorkflowError: .*asTool\('ask_nested'\) sets runConfig.modelProvider/,
      0,
    ],
    'tool-start-run-model-name': [/^nested answer$/, 2],
  };

  await worker.runUntil(async () => {
    for (const [kind, [result, modelActivities]] of Object.entries(expected)) {
      const handle = await startWorkflow(runningAgentRejectedWorkflow, {
        args: [kind as RunningAgentRejection],
        workflowExecutionTimeout: '30 seconds',
      });
      const output = await handle.result();
      t.regex(output, result, kind);
      t.notRegex(output, /the inline model ran/, kind);
      const { events } = await handle.fetchHistory();
      const scheduled = (events ?? []).filter((e) => e.activityTaskScheduledEventAttributes != null);
      t.is(scheduled.length, modelActivities, `${kind}: model Activities`);
      await replay(t, handle);
    }
  });
  // The control's change to the tool's runConfig.model reaches the nested model Activity.
  t.true(models.includes('changed-run-model'), models.join(', '));
  t.false(models.includes('nested-run-model'), models.join(', '));
});

test('Agents from another copy of @openai/agents-core are checked before they call their model', async (t) => {
  const { createWorker, taskQueue } = helpers(t);
  // Bundle a copy of the package under another name, as a dependency that ships its own copy would.
  const packageDir = fs.realpathSync(path.dirname(path.dirname(require.resolve('@openai/agents-core'))));
  const copyDir = fs.mkdtempSync(path.join(os.tmpdir(), 'agents-core-copy-'));
  t.teardown(() => fs.rmSync(copyDir, { recursive: true, force: true }));
  fs.cpSync(packageDir, copyDir, { recursive: true, filter: (src) => !/\.(d\.ts|map)$/.test(src) });
  const provider = {
    getModel() {
      return {
        async getResponse(request: any) {
          if (request.systemInstructions === 'Route.') return handoffResponse('transfer_to_desk');
          if (request.systemInstructions === 'Switch.') return toolCallResponse('switch_model', {});
          if (request.systemInstructions === 'Answer as declared.') return textResponse('declared answer');
          return textResponse('copy answer');
        },
      } as unknown as FakeModel;
    },
  };
  const plugin = new OpenAIAgentsPlugin({ modelProvider: provider });
  const workflowBundle = await bundleWorkflowCode({
    workflowsPath: require.resolve('./workflows/openai-agents-copies'),
    plugins: [plugin],
    webpackConfigHook: (config) => ({
      ...config,
      resolve: {
        ...config.resolve,
        alias: { ...(config.resolve?.alias as object), 'agents-core-copy$': path.join(copyDir, 'dist', 'index.js') },
        // The copy's own dependencies, such as `debug` and `zod`.
        modules: [...(config.resolve?.modules ?? ['node_modules']), path.dirname(path.dirname(packageDir))],
      },
    }),
  });
  const worker = await createWorker({ workflowBundle, plugins: [plugin] });
  // [expected result, model Activities the run schedules]
  const expected: Record<AgentCopyCase, [RegExp, number]> = {
    'agent-start-model-object': [/^AgentsWorkflowError: .*Agent 'Lead' has a Model object/, 0],
    'handoff-callback-model-object': [/^AgentsWorkflowError: .*Agent 'Picked' has a Model object/, 1],
    'tool-model-object': [/^AgentsWorkflowError: .*Agent 'Lead' has a Model object/, 1],
    'string-model': [/^copy answer$/, 1],
  };
  await worker.runUntil(async () => {
    for (const [kind, [result, modelActivities]] of Object.entries(expected)) {
      // By name: the Workflow module needs the bundle's `agents-core-copy`.
      const handle = await t.context.env.client.workflow.start('agentCopyWorkflow', {
        taskQueue,
        workflowId: `agent-copy-${kind}-${Date.now()}`,
        args: [kind],
        workflowExecutionTimeout: '30 seconds',
      });
      const output: string = await handle.result();
      t.regex(output, result, kind);
      t.notRegex(output, /the inline model ran/, kind);
      const { events } = await handle.fetchHistory();
      const scheduled = (events ?? []).filter((e) => e.activityTaskScheduledEventAttributes != null);
      t.is(scheduled.length, modelActivities, `${kind}: model Activities`);
      await Worker.runReplayHistory(
        { workflowBundle, plugins: [new OpenAIAgentsPlugin({ modelProvider: new FakeModelProvider([]) })] },
        await handle.fetchHistory()
      );
    }
  });
});

for (const [title, stopOnFirstTool, stream, expectCancelled] of [
  ['Cancelling during a nested model call cancels a run that stops on the tool', true, false, true],
  ['Cancelling during a nested model call cancels a streamed run that stops on the tool', true, true, true],
  ['Cancelling during a nested model call cancels a run that calls the model again', false, false, true],
  ['A run that stops on a nested tool completes when not cancelled', true, false, false],
] as const) {
  test(title, async (t) => {
    const { createWorker, startWorkflow } = helpers(t);
    let nestedStarted!: () => void;
    const nestedCalled = new Promise<void>((resolve) => (nestedStarted = resolve));
    let release!: () => void;
    const released = new Promise<void>((resolve) => (release = resolve));
    const provider = {
      getModel() {
        return {
          async getResponse(request: any) {
            if (request.systemInstructions === 'Answer as nested.') {
              nestedStarted();
              if (expectCancelled) await released;
              return textResponse('nested answer');
            }
            const toolAnswered =
              Array.isArray(request.input) && request.input.some((item: any) => item.type === 'function_call_result');
            return toolAnswered ? textResponse('lead answer') : toolCallResponse('ask_nested', { input: 'q' });
          },
          async *getStreamedResponse() {
            yield {
              type: 'response_done',
              response: {
                id: 'resp_lead',
                usage: { requests: 1, inputTokens: 10, outputTokens: 20, totalTokens: 30 },
                output: toolCallResponse('ask_nested', { input: 'q' }).output,
              },
            } as unknown as StreamEvent;
          },
        } as unknown as FakeModel;
      },
    };
    const worker = await createWorker({
      plugins: [new OpenAIAgentsPlugin({ modelProvider: provider, modelParams: { streamingTopic: 'events' } })],
    });

    await worker.runUntil(async () => {
      const handle = await startWorkflow(nestedToolCancelWorkflow, {
        args: [{ stopOnFirstTool, stream }],
        workflowExecutionTimeout: '30 seconds',
      });
      if (!expectCancelled) {
        t.is(await handle.result(), 'nested answer');
        await replay(t, handle);
        return;
      }
      // If the run ends before its nested model call, the assertions below report how.
      await Promise.race([nestedCalled, handle.result().catch(() => undefined)]);
      await handle.cancel();
      const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
      release();
      t.true(err?.cause instanceof CancelledFailure, String(err?.cause));
      const { events } = await handle.fetchHistory();
      t.is(events?.at(-1)?.eventType, EventType.EVENT_TYPE_WORKFLOW_EXECUTION_CANCELED);
      await replay(t, handle);
    });
  });
}

for (const [title, legacyFirstRun, nestedCall] of [
  ['a handoff to an agent that is also an asTool() source resumes after continueAsNew', false, 'handoff'],
  ['an approval inside a nested asTool() run resumes after continueAsNew', false, 'tool'],
  ['state saved before agent.asTool() replaced a function tool resumes', true, 'handoff'],
] as const) {
  test(title, async (t) => {
    const { createWorker, startWorkflow } = helpers(t);
    const calls: { instructions: string; model: string }[] = [];
    const answered = (request: any, callId: string) =>
      Array.isArray(request.input) &&
      request.input.some((item: any) => item.type === 'function_call_result' && item.callId === callId);
    const provider = {
      getModel(name?: string) {
        return {
          async getResponse(request: any) {
            calls.push({ instructions: request.systemInstructions, model: name ?? '' });
            if (request.systemInstructions === 'Answer as nested.') {
              return answered(request, 'call_fake_confirmAction')
                ? textResponse('nested done')
                : toolCallResponse('confirmAction', { reason: 'approved' });
            }
            if (answered(request, 'call_fake_ask_nested')) return textResponse('parent done');
            return nestedCall === 'handoff'
              ? handoffResponse('transfer_to_Nested')
              : toolCallResponse('ask_nested', { input: 'q' });
          },
        } as unknown as FakeModel;
      },
    };
    const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })] });

    await worker.runUntil(async () => {
      const handle = await startWorkflow(sharedAgentApprovalWorkflow, {
        args: [{ legacyFirstRun }],
        workflowExecutionTimeout: '30 seconds',
      });
      const result = await handle.result();
      t.is(result.savedIdentity, nestedCall === 'handoff' ? 'Nested' : 'Parent');
      t.is(result.output, nestedCall === 'handoff' ? 'nested done' : 'parent done');
      t.is(result.executions, 1, 'the approved action runs once');
      // Nested sets its own model, which wins over runConfig.model wherever it runs, as upstream.
      const nestedModels = calls.filter((c) => c.instructions === 'Answer as nested.').map((c) => c.model);
      t.deepEqual(nestedModels, ['own-model', 'own-model']);
      await replay(t, handle);
    });
  });
}

for (const [title, onAfterStart] of [
  ['A nested agent.asTool() run streams to its onStream option and on() handlers', false],
  ['An on() handler registered after the run starts gets the nested agent.asTool() stream', true],
] as const) {
  test(title, async (t) => {
    const { createWorker, startWorkflow } = helpers(t);
    let leadCalls = 0;
    const provider = {
      getModel() {
        return {
          async getResponse() {
            return leadCalls++ === 0 ? toolCallResponse('ask_streamer', { input: 'q' }) : textResponse('done');
          },
          async *getStreamedResponse() {
            yield* streamingTextEvents('streamed answer');
          },
        } as unknown as FakeModel;
      },
    };
    const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })] });

    await worker.runUntil(async () => {
      const handle = await startWorkflow(agentToolStreamWorkflow, {
        args: ['Hello', onAfterStart],
        workflowExecutionTimeout: '30 seconds',
      });
      t.deepEqual(await handle.result(), { output: 'done', onStream: ['streamed answer'], on: ['streamed answer'] });
      const { events } = await handle.fetchHistory();
      const activityTypes = (events ?? [])
        .map((e) => e.activityTaskScheduledEventAttributes?.activityType?.name)
        .filter((name) => name !== undefined);
      t.deepEqual(activityTypes, ['invokeModelActivity', 'invokeModelStreamActivity', 'invokeModelActivity']);
      await replay(t, handle);
    });
  });
}

test('A nested agent.asTool() run fails to stream when model calls are local Activities', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const leadInputs: any[] = [];
  let streamed = false;
  const provider = {
    getModel() {
      return {
        async getResponse(request: any) {
          leadInputs.push(request.input);
          return leadInputs.length === 1 ? toolCallResponse('ask_streamer', { input: 'q' }) : textResponse('done');
        },
        async *getStreamedResponse() {
          streamed = true;
          yield* streamingTextEvents('streamed answer');
        },
      } as unknown as FakeModel;
    },
  };
  const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })] });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(agentToolStreamWorkflow, {
      args: ['Hello', false, true],
      workflowExecutionTimeout: '30 seconds',
    });
    t.deepEqual(await handle.result(), { output: 'done', onStream: [], on: [] });
    t.false(streamed, 'the nested model is not called');
    // A nested run's failure is the tool's output, as upstream.
    const toolOutput = leadInputs[1].find((item: any) => item.type === 'function_call_result')?.output;
    t.regex(JSON.stringify(toolOutput), /Streaming is incompatible with useLocalActivity/);
    await replay(t, handle);
  });
});

test('agent.asTool() tools keep the options and name they were created with', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const leadTools: string[][] = [];
  const instructions: string[] = [];
  const provider = {
    getModel() {
      return {
        async getResponse(request: any) {
          instructions.push(request.systemInstructions);
          if (request.systemInstructions === 'Answer as renamed.') return textResponse('renamed answer');
          leadTools.push((request.tools ?? []).map((tool: { name: string }) => tool.name));
          return multiToolCallResponse([
            { name: 'ask_gated', args: { input: 'q1' } },
            { name: 'Original_Name', args: { input: 'q2' } },
          ]);
        },
      } as unknown as FakeModel;
    },
  };
  const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })] });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(agentToolOptionsReuseWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });
    // Reusing the options object does not drop ask_gated's approval, so it pauses instead of running.
    t.deepEqual(await handle.result(), ['ask_gated']);
    t.deepEqual(leadTools, [['ask_gated', 'ask_open', 'Original_Name']]);
    t.deepEqual(instructions, ['Use the tools.', 'Answer as renamed.']);
    await replay(t, handle);
  });
});

test('Stateful MCP: not connected produces ApplicationFailure', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('unused')]),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(statefulMcpNotConnectedWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(result, 'Stateful MCP Server not connected — did you forget to call `connect()` before using the server?');
  });
});

test('Stateful MCP: multi-run isolation under reuseV8Context', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  let callCount = 0;
  const mcpProvider = new StatefulMCPServerProvider(
    'isolationTest',
    () => {
      const marker = `server-${++callCount}`;
      const server: MCPServer = {
        cacheToolsList: false,
        name: 'isolationTest',
        async connect() {},
        async close() {},
        async listTools() {
          return [
            {
              name: 'get_marker',
              description: 'Returns marker',
              inputSchema: {
                type: 'object' as const,
                properties: {},
                required: [] as string[],
                additionalProperties: false,
              },
            },
          ];
        },
        async callTool() {
          return [{ type: 'text', text: marker }];
        },
        async invalidateToolsCache() {},
      };
      return server;
    },
    t.context.env.nativeConnection
  );

  const worker = await createWorker({
    reuseV8Context: true,
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('unused')]),
        mcpServerProviders: [mcpProvider],
      }),
    ],
  });

  await worker.runUntil(async () => {
    const resultA = await executeWorkflow(statefulMcpIsolationWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });
    const resultB = await executeWorkflow(statefulMcpIsolationWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });

    t.not(resultA, resultB, 'Two runs on the same V8 isolate must see distinct server instances');
    t.regex(resultA, /server-1/, 'First run should see server-1');
    t.regex(resultB, /server-2/, 'Second run should see server-2');
  });
});

test('Stateful MCP: heartbeat timeout produces DedicatedWorkerFailure', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const mcpProvider = new StatefulMCPServerProvider(
    'heartbeatTest',
    (): MCPServer => ({
      cacheToolsList: false,
      name: 'heartbeatTest',
      async connect() {},
      async close() {},
      async listTools(): Promise<any[]> {
        // Block longer than the 1-second heartbeatTimeout below to trigger the timeout.
        await new Promise((resolve) => setTimeout(resolve, 2_000));
        return [];
      },
      async callTool() {
        return [];
      },
      async invalidateToolsCache() {},
    }),
    t.context.env.nativeConnection
  );

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('unused')]),
        mcpServerProviders: [mcpProvider],
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(statefulMcpHeartbeatTimeoutWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.is(
      result,
      'DedicatedWorkerFailure: MCP Stateful Server Worker failed to heartbeat. Check that a worker is polling the dedicated task queue for this MCP run.'
    );
  });
});

test('Stateful MCP: slow connect heartbeat regression', async (t) => {
  // Session activity must heartbeat before server.connect() returns —
  // otherwise the periodic-only heartbeat fires too late for short timeouts.
  const { createWorker, startWorkflow } = helpers(t);

  let connectCallCount = 0;
  const mcpProvider = new StatefulMCPServerProvider(
    'slowConnectTest',
    (): MCPServer => ({
      cacheToolsList: false,
      name: 'slowConnectTest',
      async connect() {
        connectCallCount++;
        await new Promise((resolve) => setTimeout(resolve, 1500));
      },
      async close() {},
      async listTools() {
        return [
          {
            name: 'dummy',
            description: 'Dummy tool',
            inputSchema: {
              type: 'object' as const,
              properties: {},
              required: [] as string[],
              additionalProperties: false,
            },
          },
        ];
      },
      async callTool() {
        return [];
      },
      async invalidateToolsCache() {},
    }),
    t.context.env.nativeConnection
  );

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('unused')]),
        mcpServerProviders: [mcpProvider],
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(statefulMcpSlowConnectHeartbeatWorkflow, {
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.regex(result, /^connected:/, `Expected connected:N result, got: ${result}`);
    t.true(connectCallCount > 0, 'connect() should have been called');
  });
});

test('Streaming: external subscriber and in-workflow result see the same events', async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  const events = streamingTextEvents('Hello streamed world');
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new StreamingFakeModelProvider(events),
        modelParams: { streamingTopic: 'events', streamingBatchInterval: '50ms' },
      }),
    ],
  });

  // Match the modelParams the plugin uses so the config header carries streamingTopic.
  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
    interceptors: [
      new OpenAIAgentsTraceClientInterceptor({
        modelParams: { streamingTopic: 'events', streamingBatchInterval: '50ms' },
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await wfClient.start(streamingAgentWorkflow, {
      taskQueue,
      workflowId: `streaming-${Date.now()}`,
      args: ['Hi'],
      workflowExecutionTimeout: '30 seconds',
    });

    const streamClient = WorkflowStreamClient.create(t.context.env.client, handle.workflowId);
    const received: Array<{ type?: string; delta?: string }> = [];
    const gen = streamClient.topic<{ type?: string; delta?: string }>('events').subscribe(0, { pollCooldown: 0 });
    const collect = (async () => {
      for await (const item of gen) {
        received.push(item.data);
        if (received.length >= events.length) {
          await gen.return();
          break;
        }
      }
    })();

    const result = await handle.result();
    await collect;

    t.deepEqual(result.deltas, ['Hello streamed world'], 'in-workflow result yields the text deltas in order');
    t.is(result.finalOutput, 'Hello streamed world');

    t.is(received.length, events.length, 'external subscriber receives every published event in order');
    t.is(received[0]!.type, 'output_text_delta');
    t.is(received[0]!.delta, 'Hello streamed world');
    t.is(received[received.length - 1]!.type, 'response_done');
  });
});

test('Streaming: missing streamingTopic fails fast before scheduling an Activity', async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new StreamingFakeModelProvider(streamingTextEvents('unused')),
      }),
    ],
  });

  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
  });

  await worker.runUntil(async () => {
    const handle = await wfClient.start(streamingNoTopicWorkflow, {
      taskQueue,
      workflowId: `streaming-no-topic-${Date.now()}`,
      args: ['Hi'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.is(result, 'StreamingTopicNotConfigured');

    const { events } = await handle.fetchHistory();
    const modelStreamScheduled =
      events?.filter(
        (e) =>
          e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED &&
          e.activityTaskScheduledEventAttributes?.activityType?.name === 'invokeModelStreamActivity'
      ) ?? [];
    t.is(modelStreamScheduled.length, 0, 'no streaming Activity should be scheduled when the topic is unset');
  });
});

test('Streaming: useLocalActivity fails fast before scheduling an Activity', async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new StreamingFakeModelProvider(streamingTextEvents('unused')),
      }),
    ],
  });

  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
  });

  await worker.runUntil(async () => {
    const handle = await wfClient.start(streamingLocalActivityWorkflow, {
      taskQueue,
      workflowId: `streaming-local-activity-${Date.now()}`,
      args: ['Hi'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.is(result, 'StreamingLocalActivityUnsupported');
  });
});

/** Calls fetchValue, ends after the tool result, then answers a second turn with its instructions. */
function hookProbeModelProvider(): ModelProvider {
  let calls = 0;
  const model: Model = {
    async getResponse(request: ModelRequest): Promise<ModelResponse> {
      calls++;
      if (calls === 1) return toolCallResponse('fetchValue', { value: 'S-hook' });
      if (calls === 2) return textResponse('done');
      return textResponse(`next turn saw: ${request.systemInstructions}`);
    },
    // eslint-disable-next-line require-yield
    async *getStreamedResponse(): AsyncIterable<StreamEvent> {
      throw new Error('Streaming not supported');
    },
  };
  return { getModel: () => model };
}

test('agent.on listeners run and their context writes reach later turns', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);
  const plugin = new OpenAIAgentsPlugin({ modelProvider: hookProbeModelProvider() });
  const worker = await createWorker({
    plugins: [plugin],
    activities: { fetchValue: async ({ value }: { value: string }) => value },
  });

  const result = await worker.runUntil(executeWorkflow(agentHooksWorkflow, { workflowExecutionTimeout: '30 seconds' }));

  t.deepEqual(result.events, [
    'agent_start:true',
    'agent_tool_end:fetchValue',
    'agent_end:done',
    'agent_start:true',
    'agent_end:next turn saw: hook=S-hook',
  ]);
  t.is(result.nextTurnOutput, 'next turn saw: hook=S-hook');
});

const deskActivities = {
  standardDesk: async () => 'standard desk',
  vipDesk: async () => 'vip desk',
};

test('A custom Handoff hands off to the agent its callback returns', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);
  const modelProvider = new FakeModelProvider([
    handoffResponse('transfer_to_desk'),
    textResponse('vip answer'),
    handoffResponse('transfer_to_desk'),
    textResponse('standard answer'),
  ]);
  const worker = await createWorker({
    plugins: [new OpenAIAgentsPlugin({ modelProvider })],
    activities: deskActivities,
  });

  const results = await worker.runUntil(async () => [
    await executeWorkflow(customHandoffWorkflow, { args: ['vip'], workflowExecutionTimeout: '30 seconds' }),
    await executeWorkflow(customHandoffWorkflow, { args: ['standard'], workflowExecutionTimeout: '30 seconds' }),
  ]);

  t.deepEqual(results, ['vip: vip answer', 'standard: standard answer']);
});

test('A ModelSummaryProvider gets the running agent; nested agent.asTool() calls get none', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const modelProvider = new FakeModelProvider([
    toolCallResponse('ask_expert', { input: 'q' }),
    textResponse('expert answer'),
    handoffResponse('transfer_to_billing'),
    textResponse('billing done'),
  ]);
  const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider })] });
  await worker.runUntil(async () => {
    const handle = await startWorkflow(summaryAgentWorkflow, { workflowExecutionTimeout: '30 seconds' });
    t.is(await handle.result(), 'billing done');
    const { events } = await handle.fetchHistory();
    const summaries = (events ?? [])
      .filter((e) => e.activityTaskScheduledEventAttributes?.activityType?.name === 'invokeModelActivity')
      .map((e) =>
        Buffer.from(e.userMetadata?.summary?.data ?? [])
          .toString('utf-8')
          .replace(/"/g, '')
      );
    t.deepEqual(summaries, ['agent:lead', 'agent:none', 'agent:lead', 'agent:billing']);
  });
});

test('runConfig.model leaves an agent that sets its own model on that model, as upstream', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);
  const names: string[] = [];
  const provider = {
    getModel(name?: string) {
      return {
        async getResponse() {
          names.push(name ?? '');
          return textResponse('ok');
        },
      } as unknown as FakeModel;
    },
  };
  const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })] });
  await worker.runUntil(executeWorkflow(runModelOverrideWorkflow, { workflowExecutionTimeout: '30 seconds' }));
  t.deepEqual(names, ['own-model']);
});
