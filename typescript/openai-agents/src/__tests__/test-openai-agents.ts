// Integration tests that need a real Worker. Pure-function tests live in
// the sibling `*.test.ts` files.
import { setTracingDisabled } from '@openai/agents-core';
import type { MCPServer } from '@openai/agents-core';
import { WorkflowClient } from '@temporalio/client';
import { temporal } from '@temporalio/proto';
import { helpers } from '@temporalio/test-helpers';
import { WorkflowStreamClient } from '@temporalio/workflow-streams/client';
import {
  OpenAIAgentsPlugin,
  OpenAIAgentsTraceClientInterceptor,
  StatefulMCPServerProvider,
  StatelessMCPServerProvider,
} from '..';
import {
  approvalResumeContextWorkflow,
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
} from './workflows/openai-agents';
import { makeTestFunction } from './helpers/test-fn';
import { FakeModelProvider, handoffResponse, textResponse, toolCallResponse } from './stubs/openai-agents';
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

test('Approval resume keeps the caller context object', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([
          toolCallResponse('set_after', { value: 'S-1' }, 'call_set_after_1'),
          toolCallResponse('set_after', { value: 'S-2' }, 'call_set_after_2'),
          toolCallResponse('set_after', { value: 'S-after' }, 'call_set_after_3'),
          textResponse('done'),
          toolCallResponse('set_after', { value: 'A' }, 'call_shared_a'),
          toolCallResponse('set_after', { value: 'B' }, 'call_shared_b'),
          toolCallResponse('set_after', { value: 'C' }, 'call_shared_c'),
          textResponse('done A'),
          textResponse('done B'),
          textResponse('done C'),
          toolCallResponse('flaky', { value: 'R' }, 'call_flaky'),
          textResponse('done R'),
          toolCallResponse('gated', { value: 'D' }, 'call_gated_d'),
          toolCallResponse('gated', { value: 'E1' }, 'call_gated_e1'),
          toolCallResponse('gated', { value: 'E2' }, 'call_gated_e2'),
          textResponse('done E'),
          toolCallResponse('set_after', { value: 'R-1' }, 'call_rehydrated_1'),
          toolCallResponse('set_after', { value: 'R-2' }, 'call_rehydrated_2'),
          toolCallResponse('set_after', { value: 'R-3' }, 'call_rehydrated_3'),
          textResponse('done rehydrated'),
          toolCallResponse('set_after', { value: 'H-1' }, 'call_route_1'),
          handoffResponse('transfer_to_FirstTarget'),
          textResponse('done routed'),
          toolCallResponse('set_after', { value: 'H-2' }, 'call_route_2'),
          handoffResponse('transfer_to_FirstTarget'),
          toolCallResponse('extend', {}, 'call_extend'),
          toolCallResponse('extra', { value: 'X' }, 'call_extra'),
          textResponse('done extended'),
          handoffResponse('transfer_to_FailChild'),
          toolCallResponse('set_after', { value: 'F' }, 'call_fail_child'),
          textResponse('done failed conversion'),
          toolCallResponse('set_after', { value: 'V' }, 'call_versioned'),
          textResponse('done versioned'),
        ]),
      }),
    ],
  });

  const result = await worker.runUntil(
    executeWorkflow(approvalResumeContextWorkflow, { workflowExecutionTimeout: '30 seconds' })
  );

  // Each resume converts into the agent objects its state runs on, so the approvals keep one owner, and
  // no resume drops approvals that another state on the context, or a retry of the same state, needs.
  t.deepEqual(result, {
    callerAfter: 'S-after',
    resumedUsesCallerObject: true,
    writes: ['S-1', 'S-2', 'S-after'],
    approvalRounds: 3,
    failedResumeThrew: true,
    approvalOwners: 1,
    approvedCallIds: 3,
    sharedContextWrites: ['A', 'B', 'C'],
    sharedContextInterruptions: 0,
    retryThrew: true,
    retryAttempts: 2,
    retryWrites: ['R'],
    retryInterruptions: 0,
    concurrentFailedResumeThrew: true,
    concurrentWrites: ['E1', 'E2'],
    concurrentInterruptions: 0,
    rehydratedWrites: ['R-1', 'R-2', 'R-3'],
    rehydratedApprovalOwners: 2,
    rehydratedApprovedCallIds: 6,
    handoffTarget: 'SecondTarget',
    handoffFilters: ['second'],
    disabledHandoffThrew: true,
    sharedFieldsWrites: ['X'],
    sharedFieldsTemperatures: [0.1, 0.9, 0.9],
    failedConversionThrew: true,
    failedConversionWrites: ['F'],
    failedConversionInterruptions: 0,
    versions: ['before', 'after'],
    lastAgentVersion: 'after',
  });
});
