// Integration tests that need a real Worker. Pure-function tests live in
// the sibling `*.test.ts` files.
import {
  addTraceProcessor,
  createMCPToolStaticFilter,
  getCurrentTrace,
  getGlobalTraceProvider,
  setTracingDisabled,
  withCustomSpan,
  withTrace,
} from '@openai/agents-core';
import type { MCPServer, Model, ModelProvider, Span, SpanData, TracingProcessor } from '@openai/agents-core';
import { WorkflowClient, WorkflowFailedError } from '@temporalio/client';
import type { ApplicationFailure } from '@temporalio/common';
import { temporal } from '@temporalio/proto';
import { helpers } from '@temporalio/test-helpers';
import { bundleWorkflowCode } from '@temporalio/worker';
import type { NativeConnection } from '@temporalio/worker';
import { WorkflowStreamClient } from '@temporalio/workflow-streams/client';
import {
  OpenAIAgentsPlugin,
  OpenAIAgentsTraceClientInterceptor,
  StatefulMCPServerProvider,
  StatelessMCPServerProvider,
  type TracingApiKeyContext,
} from '..';
import {
  localActivityAgentWorkflow,
  mcpFactoryArgWorkflow,
  mcpToolFilterWorkflow,
  summaryOverrideStringWorkflow,
  statefulMcpNotConnectedWorkflow,
  statefulMcpIsolationWorkflow,
  statefulMcpHeartbeatTimeoutWorkflow,
  statefulMcpSlowConnectHeartbeatWorkflow,
  streamingAgentWorkflow,
  streamingLocalActivityWorkflow,
  streamingNoTopicWorkflow,
  tracingApiKeyWorkflow,
} from './workflows/openai-agents';
import { makeTestFunction } from './helpers/test-fn';
import { type FakeModel, FakeModelProvider, textResponse, toolCallResponse } from './stubs/openai-agents';
import { StreamingFakeModelProvider, streamingTextEvents } from './stubs/openai-agents-fakes';
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
      t.is((err?.cause?.cause as ApplicationFailure | undefined)?.type, 'MCPToolFilterUnsupported');
    }
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

test('tracingApiKey gives spans that cross a Temporal boundary the key the Worker resolves', async (t) => {
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
  await worker.runUntil(async () => {
    // The run sets its own key in Workflow code.
    await wfClient.execute(tracingApiKeyWorkflow, {
      taskQueue,
      workflowId: runWorkflowId,
      args: ['sk-run'],
      workflowExecutionTimeout: '30 seconds',
    });
    // The client starts the Workflow inside a trace with its own key.
    await withTrace(
      'tracing-key-client-trace',
      async () => {
        clientTraceId = getCurrentTrace()!.traceId;
        // The run's own key does not apply inside the client's trace.
        await wfClient.execute(tracingApiKeyWorkflow, {
          taskQueue,
          workflowId: traceWorkflowId,
          args: ['sk-run'],
          workflowExecutionTimeout: '30 seconds',
        });
        await wfClient.execute(statefulMcpIsolationWorkflow, {
          taskQueue,
          workflowId: mcpWorkflowId,
          workflowExecutionTimeout: '30 seconds',
        });
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

  t.deepEqual(keysIn(runTraceId, 'model-call'), ['sk-resolved-run'], 'Activity span of a run with its own key');
  t.deepEqual(keysIn(runTraceId, 'agent'), ['sk-run'], 'the run keeps its own key on its Workflow spans');
  t.deepEqual(keysIn(clientTraceId, 'model-call'), ['sk-resolved-trace'], 'Activity span in a keyed client trace');
  t.deepEqual(keysIn(clientTraceId, 'agent'), ['sk-resolved-trace'], 'Workflow span in a keyed client trace');
  t.deepEqual(keysIn(clientTraceId, 'mcp-connect'), ['sk-resolved-mcp'], 'stateful MCP span on the main Worker');
  t.deepEqual(keysIn(clientTraceId, 'mcp-list-tools'), ['sk-resolved-mcp'], 'listTools span on the dedicated Worker');
  t.deepEqual(keysIn(clientTraceId, 'mcp-call-tool'), ['sk-resolved-mcp'], 'callTool span on the dedicated Worker');
  t.like(
    contexts.find((c) => c.traceId === runTraceId && c.traceName !== undefined),
    { workflowId: runWorkflowId, workflowType: 'tracingApiKeyWorkflow' },
    'the Activity context names the Workflow that scheduled it'
  );
});
