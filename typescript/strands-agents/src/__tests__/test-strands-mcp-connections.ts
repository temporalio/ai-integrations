import type { JSONValue, McpClient } from '@strands-agents/sdk';
import type { TestFn } from 'ava';
import { ApplicationFailure } from '@temporalio/common';
import type { BaseContext } from '@temporalio/test-helpers';
import { test as anyTest, helpers } from '@temporalio/test-helpers';
import { StrandsPlugin } from '..';
import { setupTestEnv, StubMcpClient, StubModel, textTurn, toolCallTurn } from './helpers/strands';
import { failingMcpToolAgent, mcpAgent } from './workflows/strands';

const test = anyTest as TestFn<BaseContext>;

setupTestEnv(test);

test('TemporalMCPClient lists and calls tools through per-server activities', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  // Record the args the MCP client's callTool was actually invoked with, so the
  // assertion proves the per-server callTool activity ran rather than relying on
  // the stub model's scripted reply.
  const callToolArgs: Array<{ name: string; args: JSONValue }> = [];
  class CapturingMcpClient extends StubMcpClient {
    override async callTool(tool: { name: string }, args: JSONValue): Promise<JSONValue> {
      callToolArgs.push({ name: tool.name, args });
      return super.callTool(tool, args);
    }
  }
  const factory = (): McpClient =>
    new CapturingMcpClient([
      {
        name: 'listFiles',
        description: 'List files',
        inputSchema: { type: 'object', properties: { path: { type: 'string' } } },
      },
    ]);

  const worker = await createWorker({
    plugins: [
      new StrandsPlugin({
        models: {
          test: () => new StubModel([toolCallTurn('listFiles', 'call_1', { path: '/' }), textTurn('found files')]),
        },
        mcpClients: { testServer: factory },
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(mcpAgent, { args: ['list files'] });
    t.is(result, 'found files');
  });

  t.deepEqual(callToolArgs, [{ name: 'listFiles', args: { path: '/' } }]);
});

test('an MCP tool gives the model the error its callTool activity failed with', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  class FailingMcpClient extends StubMcpClient {
    override async callTool(): Promise<JSONValue> {
      throw ApplicationFailure.nonRetryable('MCP server unavailable');
    }
  }
  const factory = (): McpClient =>
    new FailingMcpClient([{ name: 'listFiles', description: 'List files', inputSchema: { type: 'object' } }]);

  const worker = await createWorker({
    plugins: [
      new StrandsPlugin({
        models: {
          test: () => new StubModel([toolCallTurn('listFiles', 'call_1', { path: '/' }), textTurn('no files')]),
        },
        mcpClients: { failingServer: factory },
      }),
    ],
  });

  await worker.runUntil(async () => {
    t.deepEqual(await executeWorkflow(failingMcpToolAgent, { args: ['list files', 'failingServer'] }), [
      { text: 'MCP server unavailable', error: 'MCP server unavailable' },
    ]);
  });
});

test('successive MCP tool calls reuse one cached worker-side connection', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const counters = { factoryCalls: 0, connects: 0, disconnects: 0, callTools: 0 };
  const callToolArgs: Array<{ name: string; args: JSONValue }> = [];
  class CountingMcpClient extends StubMcpClient {
    override async connect(): Promise<void> {
      counters.connects++;
    }
    override async disconnect(): Promise<void> {
      counters.disconnects++;
    }
    override async callTool(tool: { name: string }, args: JSONValue): Promise<JSONValue> {
      counters.callTools++;
      callToolArgs.push({ name: tool.name, args });
      return super.callTool(tool, args);
    }
  }
  const factory = (): McpClient => {
    counters.factoryCalls++;
    return new CountingMcpClient([
      { name: 'listFiles', description: 'List files', inputSchema: { type: 'object', properties: {} } },
    ]);
  };

  const worker = await createWorker({
    plugins: [
      new StrandsPlugin({
        models: {
          // Two tool-call turns in one workflow → two callTool activities.
          test: () =>
            new StubModel([
              toolCallTurn('listFiles', 'call_1', { path: '/a' }),
              toolCallTurn('listFiles', 'call_2', { path: '/b' }),
              textTurn('done'),
            ]),
        },
        mcpClients: { cachingServer: factory },
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(mcpAgent, { args: ['list files', 'cachingServer'] });
    t.is(result, 'done');
    // The initial listTools opens one connection that both tool calls then
    // reuse — a single factory/connect for the whole workflow. (No caching
    // would mean factoryCalls/connects === 3: listTools plus each call.)
    t.is(counters.callTools, 2);
    t.is(counters.factoryCalls, 1);
    t.is(counters.connects, 1);
    // The cached connection stays open mid-run; nothing has disconnected yet.
    t.is(counters.disconnects, 0);
    // Both tool calls reached the client with the model's scripted inputs.
    t.deepEqual(callToolArgs, [
      { name: 'listFiles', args: { path: '/a' } },
      { name: 'listFiles', args: { path: '/b' } },
    ]);
  });

  // Worker shutdown evicts the cached connection.
  t.is(counters.disconnects, 1);
});

test('mcpConnectionIdleTimeout evicts the cached connection while the worker runs', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const counters = { disconnects: 0 };
  class CountingMcpClient extends StubMcpClient {
    override async disconnect(): Promise<void> {
      counters.disconnects++;
    }
  }
  const factory = (): McpClient =>
    new CountingMcpClient([
      { name: 'listFiles', description: 'List files', inputSchema: { type: 'object', properties: {} } },
    ]);

  const worker = await createWorker({
    plugins: [
      new StrandsPlugin({
        models: {
          test: () => new StubModel([toolCallTurn('listFiles', 'call_1', { path: '/' }), textTurn('done')]),
        },
        mcpClients: { idleServer: factory },
        // Short idle window (duration string form) so the cached call connection
        // is evicted mid-run rather than only at worker shutdown.
        mcpConnectionIdleTimeout: '100ms',
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(mcpAgent, { args: ['list files', 'idleServer'] });
    t.is(result, 'done');
    // Wait past the short idle window: the cached connection should be
    // disconnected by the idle timer on its own. Observing a disconnect inside
    // runUntil (worker still alive) proves the eviction came from the timer, not
    // worker shutdown — with the default 5-minute window nothing would have
    // disconnected here. (The exact connect/disconnect counts aren't asserted:
    // the timer may also fire between the listTools and callTool activities,
    // forcing a reconnect, which is itself valid idle-eviction behavior.)
    await new Promise((resolve) => setTimeout(resolve, 1000));
    t.true(counters.disconnects >= 1, 'expected the idle timer to evict the connection mid-run');
  });
});

test('mcpConnectionIdleTimeout does not evict a connection with a call in flight', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const counters = { disconnects: 0, disconnectsDuringCall: 0 };
  let callInFlight = false;
  class SlowMcpClient extends StubMcpClient {
    override async disconnect(): Promise<void> {
      counters.disconnects++;
      if (callInFlight) counters.disconnectsDuringCall++;
    }
    override async callTool(tool: { name: string }, args: JSONValue): Promise<JSONValue> {
      callInFlight = true;
      try {
        // Outlast the idle window several times over.
        await new Promise((resolve) => setTimeout(resolve, 500));
        return await super.callTool(tool, args);
      } finally {
        callInFlight = false;
      }
    }
  }
  const factory = (): McpClient =>
    new SlowMcpClient([
      { name: 'listFiles', description: 'List files', inputSchema: { type: 'object', properties: {} } },
    ]);

  const worker = await createWorker({
    plugins: [
      new StrandsPlugin({
        models: {
          test: () => new StubModel([toolCallTurn('listFiles', 'call_1', { path: '/' }), textTurn('done')]),
        },
        mcpClients: { slowServer: factory },
        // Idle window far shorter than the tool call it must not interrupt.
        mcpConnectionIdleTimeout: '50ms',
      }),
    ],
  });

  await worker.runUntil(async () => {
    // The tool call outlives the idle window, so an idle timer measured from the
    // start of the call (rather than from the call completing) would disconnect
    // the transport underneath it and fail the call.
    const result = await executeWorkflow(mcpAgent, { args: ['list files', 'slowServer'] });
    t.is(result, 'done');
    t.is(counters.disconnectsDuringCall, 0, 'idle timer disconnected a connection with a call in flight');
  });
});

/**
 * MCP clients for the two-Worker tests. Each records the calls it handles and whether it was
 * disconnected during one of them; `holdNextCall` holds the next call until released.
 */
function twoWorkerMcpClients() {
  const state = {
    handledBy: [] as string[],
    disconnects: [] as string[],
    disconnectsDuringCall: [] as string[],
    hold: undefined as { started: () => void; release: Promise<void> } | undefined,
  };
  class LabeledMcpClient extends StubMcpClient {
    private callsInFlight = 0;

    constructor(private readonly label: string) {
      super([{ name: 'listFiles', description: 'List files', inputSchema: { type: 'object', properties: {} } }]);
    }
    override async disconnect(): Promise<void> {
      state.disconnects.push(this.label);
      if (this.callsInFlight > 0) state.disconnectsDuringCall.push(this.label);
    }
    override async callTool(tool: { name: string }, args: JSONValue): Promise<JSONValue> {
      state.handledBy.push(this.label);
      this.callsInFlight++;
      try {
        const hold = state.hold;
        state.hold = undefined;
        if (hold) {
          hold.started();
          await hold.release;
        }
        return await super.callTool(tool, args);
      } finally {
        this.callsInFlight--;
      }
    }
  }
  const holdNextCall = () => {
    let started!: () => void;
    let release!: () => void;
    const callStarted = new Promise<void>((resolve) => (started = resolve));
    state.hold = { started, release: new Promise<void>((resolve) => (release = resolve)) };
    return { callStarted, release };
  };
  return { state, holdNextCall, factory: (label: string) => (): McpClient => new LabeledMcpClient(label) };
}

test('two Workers with their own StrandsPlugin keep their own MCP server for one name', async (t) => {
  t.timeout(60_000);
  const { createWorker, taskQueue } = helpers(t);
  const { state, holdNextCall, factory } = twoWorkerMcpClients();
  const turns = (runs: number) =>
    Array.from({ length: runs }).flatMap(() => [toolCallTurn('listFiles', 'call_1', {}), textTurn('done')]);
  const plugin = (label: string, runs: number) =>
    new StrandsPlugin({
      models: { test: () => new StubModel(turns(runs)) },
      mcpClients: { sharedName: factory(label) },
    });
  const queues = { A: taskQueue, B: `${taskQueue}-b` };
  let runs = 0;
  const run = (queue: string) =>
    t.context.env.client.workflow.execute(mcpAgent, {
      taskQueue: queue,
      workflowId: `${queue}-${++runs}`,
      args: ['list files', 'sharedName'],
    });
  const workerA = await createWorker({ plugins: [plugin('A', 1)] });
  const workerB = await createWorker({ taskQueue: queues.B, plugins: [plugin('B', 2)] });

  await workerB.runUntil(async () => {
    const runningA = workerA.run();
    try {
      t.is(await run(queues.A), 'done');
      t.is(await run(queues.B), 'done');
      t.deepEqual(state.handledBy, ['A', 'B'], "each Worker's call must reach its own plugin's MCP server");

      // Worker A stops while Worker B's call is in flight.
      const held = holdNextCall();
      const result = run(queues.B);
      await held.callStarted;
      workerA.shutdown();
      await runningA;
      held.release();
      t.is(await result, 'done');
      t.deepEqual(state.disconnectsDuringCall, [], "Worker A's stop must not close Worker B's connection");
      t.deepEqual(state.disconnects, ['A']);
    } finally {
      if (workerA.getState() === 'RUNNING') workerA.shutdown();
      await runningA;
    }
  });
  t.deepEqual(state.disconnects, ['A', 'B']);
});

test("stopping one of two Workers that share a StrandsPlugin keeps the other's MCP connection", async (t) => {
  t.timeout(60_000);
  const { createWorker, taskQueue } = helpers(t);
  const { state, holdNextCall, factory } = twoWorkerMcpClients();
  const plugin = new StrandsPlugin({
    models: { test: () => new StubModel([toolCallTurn('listFiles', 'call_1', {}), textTurn('done')]) },
    mcpClients: { sharedPlugin: factory('S') },
  });
  const workerA = await createWorker({ plugins: [plugin] });
  const workerB = await createWorker({ taskQueue: `${taskQueue}-b`, plugins: [plugin] });

  await workerB.runUntil(async () => {
    const runningA = workerA.run();
    try {
      const held = holdNextCall();
      const result = t.context.env.client.workflow.execute(mcpAgent, {
        taskQueue: `${taskQueue}-b`,
        workflowId: `${taskQueue}-b`,
        args: ['list files', 'sharedPlugin'],
      });
      await held.callStarted;
      workerA.shutdown();
      await runningA;
      held.release();
      t.is(await result, 'done');
      t.deepEqual(state.disconnects, [], 'the connection stays open while Worker B uses the plugin');
    } finally {
      if (workerA.getState() === 'RUNNING') workerA.shutdown();
      await runningA;
    }
  });
  // The last Worker using the plugin closes it.
  t.deepEqual(state.disconnects, ['S']);
});
