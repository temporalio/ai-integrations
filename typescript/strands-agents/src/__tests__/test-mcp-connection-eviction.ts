/**
 * Unit tests for how the per-server MCP activities evict their pooled connection after a failed call.
 *
 * Runs the real `{server}-callTool` / `{server}-listTools` activities in a MockActivityEnvironment against a
 * fake `McpClient`, so no Temporal server or workflow is needed.
 */
import test from 'ava';
import type { McpClient } from '@strands-agents/sdk';
import { MockActivityEnvironment } from '@temporalio/testing';
import { _evictConnection, buildCallToolActivity, buildListToolsActivity } from '../temporal-mcp-client';
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

/** Let fire-and-forget disconnects scheduled by `releaseConnection` run. */
const flush = () => new Promise<void>((resolve) => setImmediate(resolve));

const input = (toolName: string): CallToolInput => ({ toolName, args: {} });

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
  const callTool = buildCallToolActivity(server, factory);

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

  await _evictConnection(server);
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
  const callTool = buildCallToolActivity(server, factory);
  const listTools = buildListToolsActivity(server, factory);

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
  const callTool = buildCallToolActivity(server, factory);

  await t.throwsAsync(new MockActivityEnvironment().run(callTool, input('fail')), { message: 'tool failed' });
  await flush();
  t.deepEqual(state.disconnects, ['client-1']);

  t.deepEqual(await new MockActivityEnvironment().run(callTool, input('ok')), { handledBy: 'client-2' });
  t.is(state.factoryCalls, 2);

  await _evictConnection(server);
  t.deepEqual(state.disconnects, ['client-1', 'client-2']);
});
