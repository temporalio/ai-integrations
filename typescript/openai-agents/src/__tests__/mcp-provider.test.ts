import test from 'ava';
import type { MCPServer } from '@openai/agents-core';
import { ApplicationFailure } from '@temporalio/common';
import type { NativeConnection } from '@temporalio/worker';
import { MockActivityEnvironment } from '@temporalio/testing';
import { StatelessMCPServerProvider } from '../worker/mcp-provider';
import { StatefulMCPServerProvider } from '../worker/stateful-mcp-provider';

function serverWithResolver(onConnect: () => void): MCPServer {
  return {
    cacheToolsList: false,
    name: 'tickets',
    toolMetaResolver: () => ({ tenant: 'tenant-7' }),
    async connect() {
      onConnect();
      throw new Error('connection refused');
    },
    async close() {},
    async listTools() {
      return [];
    },
    async callTool() {
      return [];
    },
    async invalidateToolsCache() {},
  };
}

test('StatelessMCPServerProvider refuses a toolMetaResolver without connecting', async (t) => {
  let connects = 0;
  const activities = new StatelessMCPServerProvider('tickets', () =>
    serverWithResolver(() => connects++)
  )._getActivities();

  for (const name of ['tickets-stateless-list-tools', 'tickets-stateless-call-tool']) {
    const err = await t.throwsAsync(activities[name]!({ toolName: 'echo', args: {} }), {
      instanceOf: ApplicationFailure,
    });
    t.is(err?.type, 'MCPToolMetaResolverUnsupported');
    t.true(err?.nonRetryable);
  }
  t.is(connects, 0);
});

test('StatefulMCPServerProvider refuses a toolMetaResolver at session start without connecting', async (t) => {
  let connects = 0;
  const activities = new StatefulMCPServerProvider(
    'tickets',
    () => serverWithResolver(() => connects++),
    {} as NativeConnection
  )._getActivities();

  const err = await t.throwsAsync(
    new MockActivityEnvironment().run(activities['tickets-stateful-server-session']!, {}),
    { instanceOf: ApplicationFailure }
  );
  t.is(err?.type, 'MCPToolMetaResolverUnsupported');
  t.true(err?.nonRetryable);
  t.is(connects, 0);
});
