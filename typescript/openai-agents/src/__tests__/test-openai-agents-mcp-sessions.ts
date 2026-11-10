import * as fs from 'node:fs';
import * as path from 'node:path';
import type { MCPServer } from '@openai/agents-core';
import { Context } from '@temporalio/activity';
import { historyFromJSON } from '@temporalio/common/lib/proto-utils';
import { temporal } from '@temporalio/proto';
import { helpers } from '@temporalio/test-helpers';
import { Worker } from '@temporalio/worker';
import { OpenAIAgentsPlugin, StatefulMCPServerProvider, DEDICATED_WORKER_FAILURE_TYPE } from '..';
import { memoryServer } from './helpers/stateful-mcp-host';
import {
  statefulMcpNotConnectedWorkflow,
  statefulMcpIsolationWorkflow,
  statefulMcpHeartbeatTimeoutWorkflow,
  statefulMcpSlowConnectHeartbeatWorkflow,
  statefulMcpSessionEndedWorkflow,
  statefulMcpOperationWaitingWorkflow,
  resumeSignal,
} from './workflows/openai-agents';
import { FakeModelProvider, textResponse } from './stubs/openai-agents';
import { makeOpenAIAgentsTestFunction } from './helpers/openai-agents';
import EventType = temporal.api.enums.v1.EventType;

const test = makeOpenAIAgentsTestFunction();

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

test('Stateful MCP: histories recorded on an earlier version replay', async (t) => {
  // Recorded on main at ea04762 with the Workflow Worker stopped, so that one
  // activation resolves a stateful tool call together with plain Activities
  // (gather) or with a Workflow cancel (cancel); and a caller's cancellation of
  // calls to two servers, one without a schedule-to-start timeout (caller-cancel).
  for (const name of [
    'stateful-mcp-gather.json',
    'stateful-mcp-gather-sts0.json',
    'stateful-mcp-cancel-sts0.json',
    'stateful-mcp-caller-cancel.json',
    // Recorded on this version, with the patch marker.
    'stateful-mcp-caller-cancel-marker.json',
  ]) {
    const file = path.resolve(__dirname, '../../src/__tests__/histories', name);
    const history = historyFromJSON(JSON.parse(fs.readFileSync(file, 'utf8')));
    await t.notThrowsAsync(
      Worker.runReplayHistory(
        {
          workflowBundle: t.context.workflowBundle,
          plugins: [new OpenAIAgentsPlugin({ modelProvider: new FakeModelProvider([]) })],
        },
        history,
        name
      ),
      name
    );
  }
});

test('Stateful MCP: an operation without a schedule-to-start timeout fails once the session ends', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const calls: string[] = [];
  const plugin = () =>
    new OpenAIAgentsPlugin({
      modelProvider: new FakeModelProvider([]),
      mcpServerProviders: [
        new StatefulMCPServerProvider(
          'sessionEnded',
          () => memoryServer('sessionEnded', 'S', calls),
          t.context.env.nativeConnection
        ),
      ],
    });

  const first = await createWorker({ plugins: [plugin()] });
  const handle = await first.runUntil(async () => {
    const handle = await startWorkflow(statefulMcpSessionEndedWorkflow, {
      args: [false],
      workflowExecutionTimeout: '40 seconds',
    });
    for (let i = 0; i < 300 && !calls.includes('put'); i++) await new Promise((r) => setTimeout(r, 100));
    return handle;
  });
  // The first Worker shut down gracefully: its session Activity completed, and
  // nothing serves the session's Task Queue any more.
  const second = await createWorker({ plugins: [plugin()] });
  await second.runUntil(async () => {
    await handle.signal(resumeSignal);
    t.deepEqual(await handle.result(), ['put=V session=S', DEDICATED_WORKER_FAILURE_TYPE]);
  });
  const { events } = await handle.fetchHistory();
  const operations = (events ?? []).filter(
    (e) =>
      e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED &&
      !e.activityTaskScheduledEventAttributes?.activityType?.name?.endsWith('-server-session')
  );
  // No operation was scheduled after the session ended.
  t.is(operations.length, 1);
});

test('Stateful MCP: an operation waiting without a schedule-to-start timeout fails when the session ends', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const calls: string[] = [];
  const plugin = () =>
    new OpenAIAgentsPlugin({
      modelProvider: new FakeModelProvider([]),
      mcpServerProviders: [
        new StatefulMCPServerProvider(
          'sessionEnded',
          () => {
            const server = memoryServer('sessionEnded', 'S', calls);
            const callTool = server.callTool.bind(server);
            // "hold" runs until its Worker shuts down; the session's Worker runs one call at a time.
            server.callTool = async (tool, args) => {
              if (tool === 'hold') {
                calls.push('hold');
                await Context.current().sleep(30_000).catch(() => {});
                return [];
              }
              return callTool(tool, args);
            };
            return server;
          },
          t.context.env.nativeConnection
        ),
      ],
    });

  const first = await createWorker({ plugins: [plugin()] });
  const handle = await first.runUntil(async () => {
    const handle = await startWorkflow(statefulMcpOperationWaitingWorkflow, { workflowExecutionTimeout: '40 seconds' });
    for (let i = 0; i < 300 && !calls.includes('hold'); i++) await new Promise((r) => setTimeout(r, 100));
    await handle.signal(resumeSignal);
    const calledTools = async () =>
      ((await handle.fetchHistory()).events ?? []).filter((e) =>
        e.activityTaskScheduledEventAttributes?.activityType?.name?.endsWith('-call-tool')
      ).length;
    for (let i = 0; i < 300 && (await calledTools()) < 2; i++) await new Promise((r) => setTimeout(r, 100));
    return handle;
  });
  // The first Worker shut down gracefully while "get" waited behind "hold": the
  // session Activity completed, and the waiting call fails without running.
  const second = await createWorker({ plugins: [plugin()] });
  await second.runUntil(async () => {
    t.deepEqual(await handle.result(), [DEDICATED_WORKER_FAILURE_TYPE]);
  });
  t.false(calls.includes('get'));
});
