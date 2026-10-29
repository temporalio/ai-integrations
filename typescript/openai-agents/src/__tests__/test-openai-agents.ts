// Integration tests that need a real Worker. Pure-function tests live in
// the sibling `*.test.ts` files.
import { setTracingDisabled } from '@openai/agents-core';
import type { MCPServer } from '@openai/agents-core';
import type { ExecutionContext } from 'ava';
import { WorkflowClient, type WorkflowHandle } from '@temporalio/client';
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
  agentToolRejectedWorkflow,
  agentToolRunConfigWorkflow,
  agentToolStreamWorkflow,
  agentToolOptionsReuseWorkflow,
  sharedAgentApprovalWorkflow,
  localActivityAgentWorkflow,
  mcpFactoryArgWorkflow,
  summaryOverrideStringWorkflow,
  statefulMcpNotConnectedWorkflow,
  statefulMcpIsolationWorkflow,
  statefulMcpHeartbeatTimeoutWorkflow,
  statefulMcpSlowConnectHeartbeatWorkflow,
  streamingAgentWorkflow,
  streamingLocalActivityWorkflow,
  streamingNoTopicWorkflow,
  type AgentToolRejection,
} from './workflows/openai-agents';
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
    'name-collision': ['AgentsWorkflowError', /different agent with the same name is a handoff target/],
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

for (const [title, legacyFirstRun, nestedCall] of [
  ['a handoff to an agent that is also an asTool() source resumes after continueAsNew', false, 'handoff'],
  ['an approval inside a nested asTool() run resumes after continueAsNew', false, 'tool'],
  ['state saved before agent.asTool() replaced a function tool resumes with the handoff run model', true, 'handoff'],
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
      // One Agent object is both the handoff target and the tool source, so both take the run model.
      const nestedModels = calls.filter((c) => c.instructions === 'Answer as nested.').map((c) => c.model);
      t.deepEqual(nestedModels, ['run-model', 'run-model']);
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
