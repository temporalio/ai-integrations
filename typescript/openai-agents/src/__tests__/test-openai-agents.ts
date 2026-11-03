// Integration tests that need a real Worker. Pure-function tests live in
// the sibling `*.test.ts` files.
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {
  addTraceProcessor,
  createMCPToolStaticFilter,
  getCurrentTrace,
  getGlobalTraceProvider,
  setTracingDisabled,
  withCustomSpan,
  withTrace,
} from '@openai/agents-core';
import type {
  MCPServer,
  Model,
  ModelProvider,
  ModelRequest,
  ModelResponse,
  Span,
  SpanData,
  StreamEvent,
  TracingProcessor,
} from '@openai/agents-core';
import type { ExecutionContext } from 'ava';
import {
  WithStartWorkflowOperation,
  WorkflowClient,
  WorkflowFailedError,
  WorkflowUpdateStage,
  type WorkflowHandle,
} from '@temporalio/client';
import { CancelledFailure, type ApplicationFailure } from '@temporalio/common';
import { temporal } from '@temporalio/proto';
import { helpers, type BaseContext } from '@temporalio/test-helpers';
import { Worker, bundleWorkflowCode } from '@temporalio/worker';
import type { NativeConnection } from '@temporalio/worker';
import { WorkflowStreamClient } from '@temporalio/workflow-streams/client';
import {
  OpenAIAgentsPlugin,
  OpenAIAgentsTraceClientInterceptor,
  StatefulMCPServerProvider,
  StatelessMCPServerProvider,
  TOOL_FILTER_UNSUPPORTED_TYPE,
  type TracingApiKeyContext,
} from '..';
import {
  agentHooksWorkflow,
  agentToolRejectedWorkflow,
  agentToolRunConfigWorkflow,
  agentToolStreamWorkflow,
  agentToolOptionsReuseWorkflow,
  copiedAgentToolApprovalWorkflow,
  customHandoffWorkflow,
  runModelOverrideWorkflow,
  sharedAgentApprovalWorkflow,
  summaryAgentWorkflow,
  implicitModelSettingsWorkflow,
  type ImplicitSettingsCase,
  localActivityAgentWorkflow,
  mcpFactoryArgWorkflow,
  mcpToolFilterResumeWorkflow,
  mcpToolFilterWorkflow,
  mcpToolMetaWorkflow,
  statefulMcpFactoryResolverWorkflow,
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
  tracingApiKeyWorkflow,
  tracingKeyRefusedWorkflow,
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
      async callTool(...callArgs: unknown[]) {
        t.deepEqual(factoryArgument, { tenantId: 'tenant-42' }, 'factoryArgument should be passed to callTool');
        t.is(callArgs.length, 2, 'callTool should get no meta argument when there is no toolMetaResolver');
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

function filterTestServer(name: string, toolFilter: MCPServer['toolFilter']): MCPServer {
  return {
    cacheToolsList: false,
    name,
    toolFilter,
    async connect() {},
    async close() {},
    async listTools() {
      return ['safe_tool', 'danger_tool', 'extra_tool'].map((toolName) => ({
        name: toolName,
        description: toolName,
        inputSchema: {
          type: 'object' as const,
          properties: {},
          required: [] as string[],
          additionalProperties: false,
        },
      }));
    },
    async callTool() {
      return [{ type: 'text', text: 'unused' }];
    },
    async invalidateToolsCache() {},
  };
}

function filterTestProviders(toolFilter: MCPServer['toolFilter'], nativeConnection: NativeConnection) {
  return [
    new StatelessMCPServerProvider('filterStateless', () => filterTestServer('filterStateless', toolFilter)),
    new StatefulMCPServerProvider('filterStateful', () => filterTestServer('filterStateful', toolFilter), nativeConnection),
  ];
}

test('MCP tool filters apply on stateless and stateful servers', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const modelProvider = new FakeModelProvider(Array.from({ length: 6 }, () => textResponse('done')));
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider,
        mcpServerProviders: filterTestProviders(
          createMCPToolStaticFilter({ blocked: ['danger_tool'] }),
          t.context.env.nativeConnection
        ),
      }),
    ],
  });

  await worker.runUntil(async () => {
    for (const [stateful, listToolsActivity] of [
      [false, 'filterStateless-stateless-list-tools'],
      [true, 'filterStateful-stateful-list-tools'],
    ] as const) {
      const handle = await startWorkflow(mcpToolFilterWorkflow, { args: [stateful], workflowExecutionTimeout: '30 seconds' });
      t.deepEqual(await handle.result(), ['done', 'done', 'done']);
      // The handle lists the server's tools once; the agent filters that list in every run.
      const { events } = await handle.fetchHistory();
      const listings = events?.filter(
        (e) => e.activityTaskScheduledEventAttributes?.activityType?.name === listToolsActivity
      );
      t.is(listings?.length, 1);
    }
  });

  // `danger_tool` is blocked by the factory server's static filter. The Workflow's filter allows
  // `extra_tool` only in the first run, whose context is admin.
  const offered = (modelProvider.getModel() as FakeModel).requests.map((r) => r.tools.map((tool) => tool.name));
  const perWorkflow = [['safe_tool', 'extra_tool'], ['safe_tool'], ['safe_tool']];
  t.deepEqual(offered, [...perWorkflow, ...perWorkflow]);
});

test('a callable toolFilter on the factory server fails the run', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([]),
        mcpServerProviders: filterTestProviders(async () => true, t.context.env.nativeConnection),
      }),
    ],
  });

  await worker.runUntil(async () => {
    for (const stateful of [false, true]) {
      const err = await t.throwsAsync(
        executeWorkflow(mcpToolFilterWorkflow, { args: [stateful], workflowExecutionTimeout: '30 seconds' }),
        { instanceOf: WorkflowFailedError }
      );
      t.is((err?.cause?.cause as ApplicationFailure | undefined)?.type, TOOL_FILTER_UNSUPPORTED_TYPE);
    }
  });
});

test('a tool filter that throws while resuming from a RunState fails the run', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([toolCallResponse('confirm', {}), textResponse('done')]),
        mcpServerProviders: filterTestProviders(undefined, t.context.env.nativeConnection),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const err = await t.throwsAsync(
      executeWorkflow(mcpToolFilterResumeWorkflow, { workflowExecutionTimeout: '15 seconds' }),
      { instanceOf: WorkflowFailedError }
    );
    t.is((err?.cause as ApplicationFailure | undefined)?.type, 'AgentsWorkflowError');
  });
});

function metaTestServer(name: string, metas: unknown[]): MCPServer {
  return {
    cacheToolsList: false,
    name,
    async connect() {},
    async close() {},
    async listTools() {
      return [
        {
          name: 'echo_meta',
          description: 'Records the request _meta',
          inputSchema: {
            type: 'object' as const,
            properties: {},
            required: [] as string[],
            additionalProperties: false,
          },
        },
      ];
    },
    async callTool(_toolName: string, _args: Record<string, unknown> | null, meta?: Record<string, unknown> | null) {
      metas.push(meta);
      return [{ type: 'text', text: 'ok' }];
    },
    async invalidateToolsCache() {},
  };
}

test('MCP toolMetaResolver sends _meta on stateless and stateful servers', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const metas: unknown[] = [];
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([
          toolCallResponse('echo_meta', {}),
          textResponse('stateless done'),
          toolCallResponse('echo_meta', {}),
          textResponse('stateful done'),
        ]),
        mcpServerProviders: [
          new StatelessMCPServerProvider('metaStateless', () => metaTestServer('metaStateless', metas)),
          new StatefulMCPServerProvider(
            'metaStateful',
            () => metaTestServer('metaStateful', metas),
            t.context.env.nativeConnection
          ),
        ],
      }),
    ],
  });

  await worker.runUntil(async () => {
    t.is(
      await executeWorkflow(mcpToolMetaWorkflow, { args: [false], workflowExecutionTimeout: '30 seconds' }),
      'stateless done'
    );
    t.is(
      await executeWorkflow(mcpToolMetaWorkflow, { args: [true], workflowExecutionTimeout: '30 seconds' }),
      'stateful done'
    );
  });

  t.deepEqual(metas, [
    { tenant: 'tenant-7', tool: 'echo_meta', server: 'metaStateless' },
    { tenant: 'tenant-7', tool: 'echo_meta', server: 'metaStateful' },
  ]);
});

test('Stateful MCP: a factory toolMetaResolver fails the operations', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([]),
        mcpServerProviders: [
          new StatefulMCPServerProvider(
            'factoryResolver',
            () => ({ ...metaTestServer('factoryResolver', []), toolMetaResolver: () => ({ tenant: 'tenant-7' }) }),
            t.context.env.nativeConnection
          ),
        ],
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(statefulMcpFactoryResolverWorkflow, { workflowExecutionTimeout: '20 seconds' });
    t.deepEqual(await handle.result(), [
      'MCPToolMetaResolverUnsupported:true',
      'MCPToolMetaResolverUnsupported:true',
      'cleanup-success',
    ]);
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
    'undefined-tool': ['AgentsWorkflowError', /Cannot read properties of undefined/],
    'undefined-handoff': ['AgentsWorkflowError', /Cannot read properties of undefined/],
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
    'tool-start-memory-session': [/^An error occurred while running the tool.*agent 'Nested' has a MemorySession/, 1],
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
          if (request.systemInstructions === 'Use the tool.') return toolCallResponse('ask_nested', { input: 'q' });
          if (request.systemInstructions === 'Answer as nested.') return textResponse('nested answer');
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
    'as-tool': [/^AgentsWorkflowError: .*tool 'ask_nested' looks like an agent.asTool\(\) tool/, 0],
    'deprecated-alias': [/^nested answer$/, 2],
    'deprecated-alias-override': [/^AgentsWorkflowError: .*tool 'ask_nested' looks like an agent.asTool\(\) tool/, 0],
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

for (const [title, stopOnFirstTool, stream, expectCancelled, maxTurns] of [
  ['Cancelling during a nested model call cancels a run that stops on the tool', true, false, true, undefined],
  ['Cancelling during a nested model call cancels a streamed run that stops on the tool', true, true, true, undefined],
  ['Cancelling during a nested model call cancels a run that calls the model again', false, false, true, undefined],
  ['Cancelling during a nested model call cancels a run that then fails', false, false, true, 1],
  ['Cancelling during a nested model call cancels a streamed run that then fails', false, true, true, 1],
  ['A run that stops on a nested tool completes when not cancelled', true, false, false, undefined],
] as const) {
  // Serial, as are the copied-tool approval tests below. Each test starts its own Worker, and on a slow
  // CI runner starting all of them at once held a Workflow Task past the sandbox's 5-second limit, which
  // crashes Node 22 ("async hook stack has become corrupted").
  test.serial(title, async (t) => {
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
        args: [{ stopOnFirstTool, stream, maxTurns }],
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

for (const kind of ['original', 'alias', 'spread', 'namespace', 'override', 'alias-override'] as const) {
  // Serial, as the nested cancellation tests above explain.
  test.serial(`an approval inside a nested asTool() run resumes on result.state (${kind} tool)`, async (t) => {
    const { createWorker, startWorkflow } = helpers(t);
    const answered = (request: any, callId: string) =>
      Array.isArray(request.input) &&
      request.input.some((item: any) => item.type === 'function_call_result' && item.callId === callId);
    const provider = {
      getModel() {
        return {
          async getResponse(request: any) {
            if (request.systemInstructions === 'Answer as nested.') {
              return answered(request, 'call_fake_confirmAction')
                ? textResponse('nested done')
                : toolCallResponse('confirmAction', { reason: 'approved' });
            }
            if (answered(request, 'call_fake_ask_nested')) return textResponse('parent done');
            const response = toolCallResponse('ask_nested', { input: 'q' });
            if (kind === 'namespace') (response.output[0] as { namespace?: string }).namespace = 'desk';
            return response;
          },
        } as unknown as FakeModel;
      },
    };
    const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })] });

    await worker.runUntil(async () => {
      const handle = await startWorkflow(copiedAgentToolApprovalWorkflow, {
        args: [kind],
        workflowExecutionTimeout: '30 seconds',
      });
      // An Agent subclass's asTool() override runs through the deprecated alias too: the parent pauses
      // on the tool before the nested run starts.
      const overridden = kind === 'override' || kind === 'alias-override';
      t.deepEqual(await handle.result(), {
        output: 'parent done',
        executions: 1,
        resumes: overridden ? 2 : 1,
        paused: overridden ? [['ask_nested'], ['confirmAction']] : [['confirmAction']],
        pending: [],
      });
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
        // Answers from the request, not a call count, so a rerun local Activity gets the same answer.
        async getResponse(request: any) {
          leadInputs.push(request.input);
          const answered =
            Array.isArray(request.input) && request.input.some((item: any) => item.type === 'function_call_result');
          return answered ? textResponse('done') : toolCallResponse('ask_streamer', { input: 'q' });
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
    const toolOutputs = leadInputs.flat().filter((item: any) => item?.type === 'function_call_result');
    t.regex(JSON.stringify(toolOutputs), /Streaming is incompatible with useLocalActivity/);
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
    // ask_early, made at module scope, is recorded, so the run isn't rejected.
    t.deepEqual(leadTools, [['ask_gated', 'ask_open', 'Original_Name', 'ask_early']]);
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

/**
 * Calls fetchValue, ends after the tool result, then answers a second turn with its instructions.
 * Each answer depends only on the request, so several Workflows can share it.
 */
function hookProbeModelProvider(): ModelProvider {
  const model: Model = {
    async getResponse(request: ModelRequest): Promise<ModelResponse> {
      const input = JSON.stringify(request.input);
      if (input.includes('again')) return textResponse(`next turn saw: ${request.systemInstructions}`);
      return input.includes('function_call_result')
        ? textResponse('done')
        : toolCallResponse('fetchValue', { value: 'S-hook' });
    },
    // eslint-disable-next-line require-yield
    async *getStreamedResponse(): AsyncIterable<StreamEvent> {
      throw new Error('Streaming not supported');
    },
  };
  return { getModel: () => model };
}

test('agent.on listeners run, their context writes reach later turns, and one that throws stops the run', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);
  const plugin = new OpenAIAgentsPlugin({ modelProvider: hookProbeModelProvider() });
  const worker = await createWorker({
    plugins: [plugin],
    activities: { fetchValue: async ({ value }: { value: string }) => value },
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(agentHooksWorkflow, { workflowExecutionTimeout: '30 seconds' });
    t.deepEqual(result.events, [
      'agent_start:true',
      'agent_tool_end:fetchValue',
      'agent_end:done',
      'agent_start:true',
      'agent_end:next turn saw: hook=S-hook',
    ]);
    t.is(result.nextTurnOutput, 'next turn saw: hook=S-hook');
    // As with Node's EventEmitter, a listener that throws fails the run.
    const err = await t.throwsAsync(
      executeWorkflow(agentHooksWorkflow, { args: [true], workflowExecutionTimeout: '30 seconds' })
    );
    t.regex(String((err as any)?.cause?.message), /a guard stopped the run/);
  });
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

test('A ModelSummaryProvider gets the agent of each model call, in nested and resumed runs too', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const modelProvider = new FakeModelProvider([
    toolCallResponse('ask_expert', { input: 'q' }),
    textResponse('expert answer'),
    handoffResponse('transfer_to_billing'),
    toolCallResponse('refund', {}),
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
    t.deepEqual(summaries, ['agent:lead', 'agent:expert', 'agent:lead', 'agent:billing', 'agent:billing']);
    await replay(t, handle);
  });
});

test.serial('With no model named, the Worker adds the model settings defaults for its own default model', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const calls: { model: string; modelSettings: unknown; reasoningEffortImplicit: unknown }[] = [];
  const provider = {
    getModel(model?: string) {
      return {
        async getResponse(request: any) {
          calls.push({
            model: model ?? '',
            modelSettings: request.modelSettings,
            reasoningEffortImplicit: request._internal?.reasoningEffortImplicit,
          });
          const answered =
            Array.isArray(request.input) && request.input.some((item: any) => item.type === 'function_call_result');
          if (request.systemInstructions === 'Use the tool.') return toolCallResponse('ask_unnamed', { input: 'q' });
          if (request.systemInstructions === 'Use echo, then answer.' && !answered) return toolCallResponse('echo', {});
          return textResponse('done');
        },
      } as unknown as FakeModel;
    },
  };
  const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })] });
  const run = async (input: ImplicitSettingsCase) => {
    const handle = await startWorkflow(implicitModelSettingsWorkflow, {
      args: [input],
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(await handle.result(), 'done');
    await replay(t, handle);
  };
  const previous = process.env.OPENAI_DEFAULT_MODEL;
  try {
    await worker.runUntil(async () => {
      // The Workflow's default model is gpt-5.4-mini. A Worker whose default isn't a GPT-5 model gets
      // none of its settings, directly or in a nested run.
      process.env.OPENAI_DEFAULT_MODEL = 'gpt-4.1';
      await run({ runSettings: { temperature: 0.3 } });
      await run({ leadModel: 'gpt-5', runSettings: { temperature: 0.3 } });
      delete process.env.OPENAI_DEFAULT_MODEL;
      await run({ runSettings: { temperature: 0.3 } });
      // The agent's own settings replace the defaults, as upstream.
      await run({ runSettings: { temperature: 0.3 }, agentSettings: { temperature: 0.9 } });
      // A nested run takes the run's settings without `toolChoice` and `parallelToolCalls`.
      await run({ leadModel: 'gpt-5', runSettings: { temperature: 0.3, toolChoice: 'required', parallelToolCalls: true } });
      // Upstream clears `toolChoice` after the tool call.
      await run({ callsTool: true, runSettings: { temperature: 0.3, toolChoice: 'required' } });
      await run({ runSettings: { reasoning: { summary: 'auto' } } });
      await run({ runSettings: { reasoning: { effort: 'low' } } });
      await run({ runSettings: { temperature: 0.3 }, clearDefaults: true });
    });
  } finally {
    if (previous === undefined) delete process.env.OPENAI_DEFAULT_MODEL;
    else process.env.OPENAI_DEFAULT_MODEL = previous;
  }
  const gpt5Defaults = { reasoning: { effort: 'none' }, text: { verbosity: 'low' } };
  t.deepEqual(calls, [
    { model: '', modelSettings: { temperature: 0.3 }, reasoningEffortImplicit: false },
    // A named model keeps upstream's settings and its marker for an implicit effort.
    {
      model: 'gpt-5',
      modelSettings: { reasoning: { effort: 'low' }, text: { verbosity: 'low' }, temperature: 0.3 },
      reasoningEffortImplicit: true,
    },
    { model: '', modelSettings: { temperature: 0.3 }, reasoningEffortImplicit: false },
    { model: '', modelSettings: { ...gpt5Defaults, temperature: 0.3 }, reasoningEffortImplicit: true },
    { model: '', modelSettings: { temperature: 0.9 }, reasoningEffortImplicit: false },
    {
      model: 'gpt-5',
      modelSettings: {
        reasoning: { effort: 'low' },
        text: { verbosity: 'low' },
        temperature: 0.3,
        toolChoice: 'required',
        parallelToolCalls: true,
      },
      reasoningEffortImplicit: true,
    },
    { model: '', modelSettings: { ...gpt5Defaults, temperature: 0.3 }, reasoningEffortImplicit: true },
    { model: '', modelSettings: { ...gpt5Defaults, temperature: 0.3, toolChoice: 'required' }, reasoningEffortImplicit: true },
    { model: '', modelSettings: { ...gpt5Defaults, temperature: 0.3 }, reasoningEffortImplicit: true },
    {
      model: '',
      modelSettings: { reasoning: { effort: 'none', summary: 'auto' }, text: { verbosity: 'low' } },
      reasoningEffortImplicit: true,
    },
    { model: '', modelSettings: { reasoning: { effort: 'low' }, text: { verbosity: 'low' } }, reasoningEffortImplicit: false },
    { model: '', modelSettings: { temperature: 0.3 }, reasoningEffortImplicit: true },
  ]);
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

// A model whose call opens a span in the model Activity, as real models do with their generation span.
class SpanningModelProvider implements ModelProvider {
  getModel(): Model {
    return {
      getResponse: () => withCustomSpan(async () => textResponse('done'), { data: { name: 'model-call' } }),
      getStreamedResponse: () => {
        throw new Error('not used');
      },
    };
  }
}

test("tracingApiKey gives spans that cross a Temporal boundary the key the Worker resolves, and a trace's own key is refused", async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  getGlobalTraceProvider().setDisabled(false);
  const ended: Array<{ traceId: string; name: string; key: string | undefined }> = [];
  const recorder: TracingProcessor = {
    onTraceStart: async () => {},
    onTraceEnd: async () => {},
    onSpanStart: async () => {},
    onSpanEnd: async (span: Span<SpanData>) => {
      const data = span.spanData;
      ended.push({
        traceId: span.traceId,
        name: data.type === 'custom' ? data.name : data.type,
        key: span.tracingApiKey,
      });
    },
    forceFlush: async () => {},
    shutdown: async () => {},
  };
  addTraceProcessor(recorder);

  const runWorkflowId = `tracing-key-run-${Date.now()}`;
  const traceWorkflowId = `tracing-key-trace-${Date.now()}`;
  const mcpWorkflowId = `tracing-key-mcp-${Date.now()}`;
  const contexts: TracingApiKeyContext[] = [];
  // connect() runs in the session Activity on the main Worker; listTools() and
  // callTool() run on the dedicated per-run Worker.
  const mcpProvider = new StatefulMCPServerProvider(
    'isolationTest',
    (): MCPServer => ({
      cacheToolsList: false,
      name: 'isolationTest',
      connect: () => withCustomSpan(async () => {}, { data: { name: 'mcp-connect' } }),
      async close() {},
      listTools: () =>
        withCustomSpan(
          async () => [
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
          ],
          { data: { name: 'mcp-list-tools' } }
        ),
      callTool: () =>
        withCustomSpan(async () => [{ type: 'text', text: 'marker' }], { data: { name: 'mcp-call-tool' } }),
      async invalidateToolsCache() {},
    }),
    (t.context as any).env.nativeConnection
  );
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new SpanningModelProvider(),
        mcpServerProviders: [mcpProvider],
        tracingApiKey: (context) => {
          contexts.push(context);
          if (context.workflowId === runWorkflowId) return 'sk-resolved-run';
          if (context.workflowId === traceWorkflowId) return 'sk-resolved-trace';
          if (context.workflowId === mcpWorkflowId) return 'sk-resolved-mcp';
          return undefined;
        },
      }),
    ],
  });
  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
    interceptors: [new OpenAIAgentsTraceClientInterceptor()],
  });

  let clientTraceId = '';
  const refusal = "The OpenAI Agents plugin cannot pass a trace's own tracing key";
  await worker.runUntil(async () => {
    // The run starts its own trace.
    await wfClient.execute(tracingApiKeyWorkflow, {
      taskQueue,
      workflowId: runWorkflowId,
      workflowExecutionTimeout: '30 seconds',
    });
    // The client starts the Workflows inside its trace.
    await withTrace('tracing-key-client-trace', async () => {
      clientTraceId = getCurrentTrace()!.traceId;
      await wfClient.execute(tracingApiKeyWorkflow, {
        taskQueue,
        workflowId: traceWorkflowId,
        workflowExecutionTimeout: '30 seconds',
      });
      await wfClient.execute(statefulMcpIsolationWorkflow, {
        taskQueue,
        workflowId: mcpWorkflowId,
        workflowExecutionTimeout: '30 seconds',
      });
    });

    // A trace's own key is refused in Workflow code before the model call, and at any other Temporal call.
    for (const entryPoint of [
      'run',
      'stream',
      'runConfig',
      'withTrace',
      'withTraceOnly',
      'traceContext',
      'activity',
    ] as const) {
      const handle = await wfClient.start(tracingKeyRefusedWorkflow, {
        taskQueue,
        workflowId: `tracing-key-refused-${entryPoint}-${Date.now()}`,
        args: [entryPoint],
        workflowExecutionTimeout: '30 seconds',
      });
      const result = await handle.result();
      t.true(result.startsWith(`refused: ${refusal}`), `${entryPoint}: ${result}`);
      const { events } = await handle.fetchHistory();
      const scheduled = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED) ?? [];
      t.is(scheduled.length, 0, `${entryPoint}: no Activity`);
    }
    // Uncaught, the refusal fails the Workflow Task, which is retried until the code changes.
    const uncaught = await wfClient.start(tracingApiKeyWorkflow, {
      taskQueue,
      workflowId: `tracing-key-uncaught-${Date.now()}`,
      args: ['sk-run'],
      workflowExecutionTimeout: '30 seconds',
    });
    let taskFailure: string | null | undefined;
    for (let i = 0; i < 50 && taskFailure == null; i++) {
      const { events } = await uncaught.fetchHistory();
      taskFailure = events?.find((e) => e.eventType === EventType.EVENT_TYPE_WORKFLOW_TASK_FAILED)
        ?.workflowTaskFailedEventAttributes?.failure?.message;
      if (taskFailure == null) await new Promise((resolve) => setTimeout(resolve, 100));
    }
    t.true(taskFailure?.startsWith(refusal), `an uncaught refusal fails the Workflow Task: ${taskFailure}`);
    await uncaught.terminate();
    // The client is refused when it calls a Workflow inside a trace with its own key.
    const keyedOptions = {
      taskQueue,
      workflowId: `tracing-key-client-key-${Date.now()}`,
      workflowExecutionTimeout: '30 seconds' as const,
    };
    const keyedHandle = wfClient.getHandle(keyedOptions.workflowId);
    const clientCalls = {
      start: () => wfClient.start(tracingApiKeyWorkflow, keyedOptions),
      signal: () => keyedHandle.signal('unused'),
      query: () => keyedHandle.query('unused'),
      startUpdate: () => keyedHandle.startUpdate('unused', { waitForStage: WorkflowUpdateStage.ACCEPTED }),
      signalWithStart: () => wfClient.signalWithStart(tracingApiKeyWorkflow, { ...keyedOptions, signal: 'unused' }),
      startUpdateWithStart: () =>
        wfClient.startUpdateWithStart('unused', {
          waitForStage: WorkflowUpdateStage.ACCEPTED,
          startWorkflowOperation: new WithStartWorkflowOperation(tracingApiKeyWorkflow, {
            ...keyedOptions,
            workflowIdConflictPolicy: 'FAIL',
          }),
        }),
    };
    await withTrace(
      'tracing-key-keyed-client-trace',
      async () => {
        for (const [name, call] of Object.entries(clientCalls)) {
          await t.throwsAsync(call, { message: (message) => message.startsWith(refusal) }, name);
        }
      },
      { tracingApiKey: 'sk-client' }
    );
  });

  const runTraceId = contexts.find((c) => c.workflowId === runWorkflowId && c.traceName !== undefined)?.traceId;
  t.truthy(runTraceId, 'the resolver should be asked about the run trace in the model Activity');
  const keysIn = (traceId: string | undefined, name: string) =>
    ended.filter((s) => s.traceId === traceId && s.name === name).map((s) => s.key);
  // Workflow spans reach the Worker through a sink after each Workflow Task, so wait for them.
  for (let i = 0; i < 50 && keysIn(clientTraceId, 'agent').length === 0; i++) {
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  t.deepEqual(keysIn(runTraceId, 'model-call'), ['sk-resolved-run'], 'Activity span of a run');
  t.deepEqual(keysIn(runTraceId, 'agent'), ['sk-resolved-run'], 'Workflow span of a run');
  t.deepEqual(keysIn(clientTraceId, 'model-call'), ['sk-resolved-trace'], 'Activity span in a client trace');
  t.deepEqual(keysIn(clientTraceId, 'agent'), ['sk-resolved-trace'], 'Workflow span in a client trace');
  t.deepEqual(keysIn(clientTraceId, 'mcp-connect'), ['sk-resolved-mcp'], 'stateful MCP span on the main Worker');
  t.deepEqual(keysIn(clientTraceId, 'mcp-list-tools'), ['sk-resolved-mcp'], 'listTools span on the dedicated Worker');
  t.deepEqual(keysIn(clientTraceId, 'mcp-call-tool'), ['sk-resolved-mcp'], 'callTool span on the dedicated Worker');
  t.like(
    contexts.find((c) => c.traceId === runTraceId && c.traceName !== undefined),
    { workflowId: runWorkflowId, workflowType: 'tracingApiKeyWorkflow' },
    'the Activity context names the Workflow that scheduled it'
  );
});
