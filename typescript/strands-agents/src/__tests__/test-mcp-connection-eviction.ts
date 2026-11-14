/**
 * Unit tests for how the per-server MCP activities evict their pooled connection after a failed call
 * and when the last Worker stops.
 *
 * Runs the real `{server}-callTool` / `{server}-listTools` activities in a MockActivityEnvironment against a
 * fake `McpClient`, so no Temporal server or workflow is needed.
 */
import test from 'ava';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import type { McpClient } from '@strands-agents/sdk';
import { ApplicationFailure } from '@temporalio/common';
import { MockActivityEnvironment } from '@temporalio/testing';
import { StrandsPlugin } from '..';
import { buildCallToolActivity, buildListToolsActivity, McpConnections } from '../temporal-mcp-client';
import type { CallToolInput } from '../temporal-mcp-client';

/**
 * Fake MCP client factory. Every client it creates has an id (`client-1`, `client-2`, ...) and records
 * `disconnect()` calls in `disconnects`. Like the real MCP SDK, `disconnect()` rejects calls still in flight on
 * that client, so closing a connection under an active call surfaces as "Connection closed".
 */
function fakeMcpClients(behavior: {
  callTool: (clientId: string, toolName: string) => Promise<unknown>;
  listTools?: (clientId: string) => Promise<unknown[]>;
}) {
  const state = { factoryCalls: 0, disconnects: [] as string[] };
  const factory = (): McpClient => {
    const id = `client-${++state.factoryCalls}`;
    let closed = false;
    let rejectInflight: (err: Error) => void = () => {};
    const closedPromise = new Promise<never>((_, reject) => {
      rejectInflight = reject;
    });
    closedPromise.catch(() => {});
    return {
      async connect() {},
      async disconnect() {
        closed = true;
        state.disconnects.push(id);
        rejectInflight(new Error('Connection closed'));
      },
      async callTool(tool: { name: string }) {
        if (closed) throw new Error('Connection closed');
        return Promise.race([behavior.callTool(id, tool.name), closedPromise]);
      },
      async listTools() {
        if (closed) throw new Error('Connection closed');
        return Promise.race([behavior.listTools?.(id) ?? Promise.resolve([]), closedPromise]);
      },
    } as unknown as McpClient;
  };
  return { factory, state };
}

function deferred<T = void>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

/** Let fire-and-forget disconnects scheduled by `McpConnections.release` run. */
const flush = () => new Promise<void>((resolve) => setImmediate(resolve));

const input = (toolName: string): CallToolInput => ({ toolName, args: {} });

for (const operation of ['listTools', 'callTool']) {
  test(`${operation} rejects invalid MCP factory results before connecting`, async (t) => {
    const calls: string[] = [];
    const methods = ['connect', 'disconnect', 'listTools', 'callTool'] as const;
    const client = Object.fromEntries(methods.map((method) => [method, async () => calls.push(method)]));
    const invalidResults: [unknown, string][] = [
      [undefined, 'undefined'],
      [null, 'null'],
      [{}, 'Object'],
      ...methods.map((method): [unknown, string] => [{ ...client, [method]: undefined }, 'Object']),
      ...methods.map((method): [unknown, string] => [{ ...client, [method]: 1 }, 'Object']),
      // The MCP SDK's Client, which has close() in place of disconnect().
      [new Client({ name: 'test-client', version: '1.0.0' }), 'Client'],
    ];
    let result: unknown;
    let factoryCalls = 0;
    const factory = (): McpClient => {
      factoryCalls++;
      return result as McpClient;
    };
    const connections = new McpConnections();
    const activity: (args: CallToolInput & { server: string }) => Promise<unknown> = operation === 'listTools'
      ? buildListToolsActivity('invalid', factory, connections)
      : buildCallToolActivity('invalid', factory, connections);
    for (const [value, kind] of invalidResults) {
      // Passed in place of the factory, the value fails when the Worker is built.
      t.throws(() => new StrandsPlugin({ mcpClients: { invalid: value as () => McpClient } }), {
        instanceOf: TypeError,
        message: `MCP client factory 'invalid' is ${kind}, not a function.`,
      });

      result = value;
      const err = await t.throwsAsync(new MockActivityEnvironment().run(activity, { ...input('echo'), server: 'invalid' }), {
        instanceOf: ApplicationFailure,
      });
      t.true(err!.nonRetryable);
      t.is(err!.type, 'MCPClientFactoryTypeError');
      t.is(
        err!.message,
        `MCP client factory 'invalid' returned ${kind}, not a client with connect(), disconnect(), listTools() and callTool() methods.`
      );
    }
    t.is(factoryCalls, invalidResults.length);
    t.deepEqual(calls, []);
  });

  test(`${operation} preserves an MCP factory exception`, async (t) => {
    const failure = new Error('factory failed');
    let factoryCalls = 0;
    const factory = (): McpClient => {
      factoryCalls++;
      throw failure;
    };
    const connections = new McpConnections();
    const activity: (args: CallToolInput & { server: string }) => Promise<unknown> = operation === 'listTools'
      ? buildListToolsActivity('failure', factory, connections)
      : buildCallToolActivity('failure', factory, connections);
    for (let attempt = 0; attempt < 2; attempt++) {
      t.is(await t.throwsAsync(new MockActivityEnvironment().run(activity, { ...input('echo'), server: 'failure' })), failure);
    }
    t.is(factoryCalls, 2);
  });
}

test('a failed callTool does not close the shared connection under an in-flight sibling call', async (t) => {
  const server = 'siblingServer';
  const slowStarted = deferred();
  const releaseSlow = deferred();
  const { factory, state } = fakeMcpClients({
    callTool: async (clientId, toolName) => {
      if (toolName === 'slow') {
        slowStarted.resolve();
        await releaseSlow.promise;
        return { handledBy: clientId };
      }
      if (toolName === 'fail') {
        await slowStarted.promise;
        throw new Error('tool failed');
      }
      return { handledBy: clientId };
    },
  });
  const connections = new McpConnections();
  const callTool = buildCallToolActivity(server, factory, connections);

  const slowCall = new MockActivityEnvironment().run(callTool, input('slow'));
  const failingCall = new MockActivityEnvironment().run(callTool, input('fail'));

  await t.throwsAsync(failingCall, { message: 'tool failed' });
  await flush();
  t.deepEqual(state.disconnects, [], 'the connection must stay open while the slow call is in flight');
  t.is(state.factoryCalls, 1, 'both concurrent calls should have shared one connection');

  // The failed connection is not reused: a call made after the failure opens a fresh one.
  t.deepEqual(await new MockActivityEnvironment().run(callTool, input('ok')), { handledBy: 'client-2' });
  t.is(state.factoryCalls, 2);
  t.deepEqual(state.disconnects, [], 'the broken connection must stay open until the slow call drains');

  releaseSlow.resolve();
  t.deepEqual(await slowCall, { handledBy: 'client-1' }, 'the slow call must still succeed');
  await flush();
  t.deepEqual(state.disconnects, ['client-1'], 'the broken connection is closed once its last call finishes');

  await connections.evict(server);
  t.deepEqual(state.disconnects, ['client-1', 'client-2']);
});

test('a failed listTools does not close the shared connection under an in-flight callTool', async (t) => {
  const server = 'listFailServer';
  const slowStarted = deferred();
  const releaseSlow = deferred();
  const { factory, state } = fakeMcpClients({
    callTool: async (clientId) => {
      slowStarted.resolve();
      await releaseSlow.promise;
      return { handledBy: clientId };
    },
    listTools: async () => {
      await slowStarted.promise;
      throw new Error('list failed');
    },
  });
  const connections = new McpConnections();
  const callTool = buildCallToolActivity(server, factory, connections);
  const listTools = buildListToolsActivity(server, factory, connections);

  const slowCall = new MockActivityEnvironment().run(callTool, input('slow'));
  await t.throwsAsync(new MockActivityEnvironment().run(listTools, { server }), { message: 'list failed' });
  await flush();
  t.deepEqual(state.disconnects, [], 'the connection must stay open while the callTool is in flight');

  releaseSlow.resolve();
  t.deepEqual(await slowCall, { handledBy: 'client-1' });
  await flush();
  t.deepEqual(state.disconnects, ['client-1']);
  t.is(state.factoryCalls, 1);
});

test('a failed callTool with nothing else in flight closes the connection and the next call reconnects', async (t) => {
  const server = 'soloServer';
  const { factory, state } = fakeMcpClients({
    callTool: async (clientId, toolName) => {
      if (toolName === 'fail') throw new Error('tool failed');
      return { handledBy: clientId };
    },
  });
  const connections = new McpConnections();
  const callTool = buildCallToolActivity(server, factory, connections);

  await t.throwsAsync(new MockActivityEnvironment().run(callTool, input('fail')), { message: 'tool failed' });
  await flush();
  t.deepEqual(state.disconnects, ['client-1']);

  t.deepEqual(await new MockActivityEnvironment().run(callTool, input('ok')), { handledBy: 'client-2' });
  t.is(state.factoryCalls, 2);

  await connections.evict(server);
  t.deepEqual(state.disconnects, ['client-1', 'client-2']);
});

test('the last Worker to stop closes a connection only after its in-flight call finishes', async (t) => {
  const server = 'lastWorkerServer';
  const slowStarted = deferred();
  const releaseSlow = deferred();
  const { factory, state } = fakeMcpClients({
    callTool: async (clientId) => {
      slowStarted.resolve();
      await releaseSlow.promise;
      return { handledBy: clientId };
    },
  });
  const connections = new McpConnections();
  const callTool = buildCallToolActivity(server, factory, connections);
  const workerStopped = deferred();
  const worker = connections.use(() => workerStopped.promise);

  const slowCall = new MockActivityEnvironment().run(callTool, input('slow'));
  await slowStarted.promise;
  workerStopped.resolve();
  await worker;
  t.deepEqual(state.disconnects, [], 'the connection must stay open while its call is in flight');

  releaseSlow.resolve();
  t.deepEqual(await slowCall, { handledBy: 'client-1' });
  await flush();
  t.deepEqual(state.disconnects, ['client-1'], 'the connection is closed once its call finishes');
});
