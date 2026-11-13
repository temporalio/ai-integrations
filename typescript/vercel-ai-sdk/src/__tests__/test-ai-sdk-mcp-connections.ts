import { jsonSchema } from 'ai';
import type { MCPClient } from '@ai-sdk/mcp';
import test from 'ava';
import { ApplicationFailure } from '@temporalio/common';
import { assertEventually } from '@temporalio/test-helpers';
import { createActivities } from '..';
import { helloWorkflowGenerator, TestProvider } from './stubs/ai-sdk';

for (const idleMs of [0, 60_000]) {
  for (const operation of ['listTools', 'callTool']) {
    test(`${operation} rejects invalid MCP clients without retries (idle timeout ${idleMs})`, async (t) => {
      const invalidClients: unknown[] = [
        undefined,
        null,
        42,
        {},
        { tools: async () => ({}) },
        { close: async () => {} },
        { tools: 1, close: async () => {} },
        { tools: async () => ({}), close: 1 },
      ];
      let client: unknown;
      let factoryCalls = 0;
      const name = `invalid-${operation}-${idleMs}`;
      const activities = createActivities(
        new TestProvider(helloWorkflowGenerator()),
        {
          [name]: async () => {
            factoryCalls++;
            return client as MCPClient;
          },
        },
        { mcpConnectionIdleTimeout: idleMs }
      ) as Record<string, (args: unknown) => Promise<unknown>>;

      for (client of invalidClients) {
        const err = await t.throwsAsync(
          activities[`${name}-${operation}`]!({ name: 'testTool', input: {}, options: {} }),
          { instanceOf: ApplicationFailure }
        );
        t.true(err!.nonRetryable);
        t.is(err!.type, 'MCPClientFactoryTypeError');
        t.is(err!.message, `MCP client factory '${name}' must return a client with tools() and close() methods.`);
      }
      t.is(factoryCalls, invalidClients.length);
    });

    test(`${operation} preserves MCP factory failures (idle timeout ${idleMs})`, async (t) => {
      const failure = new Error('connection refused');
      let factoryCalls = 0;
      const name = `failed-${operation}-${idleMs}`;
      const activities = createActivities(
        new TestProvider(helloWorkflowGenerator()),
        {
          [name]: async () => {
            factoryCalls++;
            throw failure;
          },
        },
        { mcpConnectionIdleTimeout: idleMs }
      ) as Record<string, (args: unknown) => Promise<unknown>>;

      for (let attempt = 0; attempt < 2; attempt++) {
        const err = await t.throwsAsync(activities[`${name}-${operation}`]!({}));
        t.is(err, failure);
      }
      t.is(factoryCalls, 2);
    });
  }
}

test('callToolActivity awaits tool.execute before closing MCP client', async (t) => {
  // Track the order of operations to verify close() is called AFTER execute() completes
  const operationOrder: string[] = [];

  // Create a mock MCP client that tracks when close() is called
  // We only need to implement the methods used by callToolActivity: tools() and close()
  const mockMcpClient = {
    async tools() {
      return {
        testTool: {
          description: 'A test tool',
          inputSchema: { type: 'object' },
          execute: async (_input: unknown, _options: unknown) => {
            operationOrder.push('execute-start');
            // Simulate async work - without await in the activity, close() will be called before this resolves
            await new Promise((resolve) => setTimeout(resolve, 50));
            operationOrder.push('execute-end');
            return { result: 'success' };
          },
        },
      };
    },
    async close() {
      operationOrder.push('close');
    },
  };

  // Create a factory that returns the mock client - cast to any since we only implement the methods we need

  const mockMcpClientFactory = async () => mockMcpClient as any;

  // Create activities with the mock MCP client factory. mcpConnectionIdleTimeout: 0 opts out of
  // connection reuse so this test's create-then-close-per-call assertion still holds.
  const activities = createActivities(
    new TestProvider(helloWorkflowGenerator()),
    { testServer: mockMcpClientFactory },
    { mcpConnectionIdleTimeout: 0 }
  ) as Record<string, (args: unknown) => Promise<unknown>>;

  // Get the callTool activity
  const callToolActivity = activities['testServer-callTool']!;
  t.truthy(callToolActivity, 'callToolActivity should exist');

  // Call the activity
  const result = await callToolActivity({
    name: 'testTool',
    input: {},
    options: {},
  });

  // Verify the result
  t.deepEqual(result, { result: 'success' });

  // Verify the order: execute should complete BEFORE close is called
  t.deepEqual(
    operationOrder,
    ['execute-start', 'execute-end', 'close'],
    'close() should be called after execute() completes, not before'
  );
});

test('MCP client connection is reused across repeated tools()/callTool invocations within the idle window', async (t) => {
  let factoryCalls = 0;
  const closeCalls: string[] = [];

  const mockMcpClientFactory = async () => {
    factoryCalls++;
    const clientId = `client-${factoryCalls}`;
    return {
      async tools() {
        return {
          testTool: {
            description: 'A test tool',
            // The real @ai-sdk/mcp client always normalizes tool input schemas into a Schema
            // object via `jsonSchema()` before returning them from `tools()`; a bare JSON
            // Schema literal (e.g. `{ type: 'object' }`) is not a valid `FlexibleSchema` and
            // makes `asSchema()` in `extractTools` (used by `listToolsActivity`) throw.
            inputSchema: jsonSchema({ type: 'object' }),
            execute: async () => ({ result: clientId }),
          },
        };
      },
      async close() {
        closeCalls.push(clientId);
      },
    } as any;
  };

  const activities = createActivities(
    new TestProvider(helloWorkflowGenerator()),
    { reuseServer: mockMcpClientFactory },
    { mcpConnectionIdleTimeout: '1 minute' }
  ) as Record<string, (args: unknown) => Promise<unknown>>;

  const listToolsActivity = activities['reuseServer-listTools']!;
  const callToolActivity = activities['reuseServer-callTool']!;

  await listToolsActivity({});
  const firstResult = await callToolActivity({ name: 'testTool', input: {}, options: {} });
  const secondResult = await callToolActivity({ name: 'testTool', input: {}, options: {} });

  t.is(factoryCalls, 1, 'mcpClientFactory should only be called once while the connection is reused');
  t.deepEqual(firstResult, { result: 'client-1' });
  t.deepEqual(secondResult, { result: 'client-1' });
  t.deepEqual(closeCalls, [], 'the connection should not be closed while still within the idle window');
});

test('idle MCP client connection is closed and evicted after the idle timeout elapses', async (t) => {
  let factoryCalls = 0;
  const closeCalls: string[] = [];

  const mockMcpClientFactory = async () => {
    factoryCalls++;
    const clientId = `client-${factoryCalls}`;
    return {
      async tools() {
        return {
          testTool: {
            description: 'A test tool',
            inputSchema: { type: 'object' },
            execute: async () => ({ result: clientId }),
          },
        };
      },
      async close() {
        closeCalls.push(clientId);
      },
    } as any;
  };

  const activities = createActivities(
    new TestProvider(helloWorkflowGenerator()),
    { idleEvictServer: mockMcpClientFactory },
    { mcpConnectionIdleTimeout: 20 }
  ) as Record<string, (args: unknown) => Promise<unknown>>;

  const callToolActivity = activities['idleEvictServer-callTool']!;

  await callToolActivity({ name: 'testTool', input: {}, options: {} });
  t.is(factoryCalls, 1);
  t.deepEqual(closeCalls, [], 'connection should still be open immediately after the call');

  // Timer callbacks can be delayed on busy runners; wait for eviction instead of a fixed sleep.
  await assertEventually(t, (t) => {
    t.deepEqual(closeCalls, ['client-1'], 'connection should be closed once the idle timeout elapses');
  });

  // A subsequent call should reconnect via a fresh factory call.
  await callToolActivity({ name: 'testTool', input: {}, options: {} });
  t.is(factoryCalls, 2, 'a new connection should be created after eviction');
});

test('a long-running call is not evicted mid-call by a short idle timeout', async (t) => {
  let factoryCalls = 0;
  const closeCalls: string[] = [];

  const mockMcpClientFactory = async () => {
    factoryCalls++;
    const clientId = `client-${factoryCalls}`;
    return {
      async tools() {
        return {
          testTool: {
            description: 'A test tool',
            inputSchema: { type: 'object' },
            execute: async () => {
              // Run well past the configured idle timeout while the call is in flight.
              await new Promise((resolve) => setTimeout(resolve, 100));
              return { result: clientId };
            },
          },
        };
      },
      async close() {
        closeCalls.push(clientId);
      },
    } as any;
  };

  const activities = createActivities(
    new TestProvider(helloWorkflowGenerator()),
    { longCallServer: mockMcpClientFactory },
    { mcpConnectionIdleTimeout: 20 }
  ) as Record<string, (args: unknown) => Promise<unknown>>;

  const callToolActivity = activities['longCallServer-callTool']!;

  const result = await callToolActivity({ name: 'testTool', input: {}, options: {} });

  t.deepEqual(result, { result: 'client-1' });
  t.deepEqual(closeCalls, [], 'the connection must not be closed by the idle timer while the call is still in flight');
  t.is(factoryCalls, 1, 'the in-flight call should not have triggered a reconnect');
});

test('a tool-level error on one concurrent call does not evict the connection out from under a sibling call', async (t) => {
  let factoryCalls = 0;
  const closeCalls: string[] = [];
  let markHealthyCallStarted!: () => void;
  const healthyCallStarted = new Promise<void>((resolve) => {
    markHealthyCallStarted = resolve;
  });
  let releaseHealthyCall!: () => void;
  const healthyCallCanFinish = new Promise<void>((resolve) => {
    releaseHealthyCall = resolve;
  });

  const mockMcpClientFactory = async () => {
    factoryCalls++;
    const clientId = `client-${factoryCalls}`;
    return {
      async tools() {
        return {
          failingTool: {
            description: 'A tool that always fails',
            inputSchema: { type: 'object' },
            execute: async () => {
              await healthyCallStarted;
              throw new Error('business logic failure, unrelated to the connection');
            },
          },
          slowTool: {
            description: 'A slow but healthy tool',
            inputSchema: { type: 'object' },
            execute: async () => {
              markHealthyCallStarted();
              await healthyCallCanFinish;
              return { result: clientId };
            },
          },
        };
      },
      async close() {
        closeCalls.push(clientId);
      },
    } as any;
  };

  const activities = createActivities(
    new TestProvider(helloWorkflowGenerator()),
    { sharedServer: mockMcpClientFactory },
    { mcpConnectionIdleTimeout: '1 minute' }
  ) as Record<string, (args: unknown) => Promise<unknown>>;

  const callToolActivity = activities['sharedServer-callTool']!;

  const failingCall = callToolActivity({ name: 'failingTool', input: {}, options: {} });
  const healthyCall = callToolActivity({ name: 'slowTool', input: {}, options: {} });

  await t.throwsAsync(failingCall, { message: /business logic failure/ });
  const closeCallsWhileSiblingInFlight = [...closeCalls];
  releaseHealthyCall();

  const healthyResult = await healthyCall;
  t.deepEqual(
    closeCallsWhileSiblingInFlight,
    [],
    'the shared connection must not be closed while the sibling call is still in flight'
  );
  t.deepEqual(healthyResult, { result: 'client-1' }, 'the sibling call should complete successfully');
  t.is(factoryCalls, 1, 'both concurrent calls should have shared the same pooled connection');
});

test('mcpConnectionIdleTimeout: 0 opts out of connection reuse (create-then-close every call)', async (t) => {
  let factoryCalls = 0;
  const closeCalls: string[] = [];

  const mockMcpClientFactory = async () => {
    factoryCalls++;
    const clientId = `client-${factoryCalls}`;
    return {
      async tools() {
        return {
          testTool: {
            description: 'A test tool',
            inputSchema: { type: 'object' },
            execute: async () => ({ result: clientId }),
          },
        };
      },
      async close() {
        closeCalls.push(clientId);
      },
    } as any;
  };

  const activities = createActivities(
    new TestProvider(helloWorkflowGenerator()),
    { optOutServer: mockMcpClientFactory },
    { mcpConnectionIdleTimeout: 0 }
  ) as Record<string, (args: unknown) => Promise<unknown>>;

  const callToolActivity = activities['optOutServer-callTool']!;

  await callToolActivity({ name: 'testTool', input: {}, options: {} });
  await callToolActivity({ name: 'testTool', input: {}, options: {} });

  t.is(factoryCalls, 2, 'a fresh client should be created for every call when reuse is disabled');
  t.deepEqual(closeCalls, ['client-1', 'client-2'], 'each client should be closed immediately after its call');
});
