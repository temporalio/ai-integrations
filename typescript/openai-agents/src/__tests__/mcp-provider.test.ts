import test from 'ava';
import type { MCPServer } from '@openai/agents-core';
import { ApplicationFailure } from '@temporalio/common';
import { StatelessMCPServerProvider } from '../worker/mcp-provider';

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
