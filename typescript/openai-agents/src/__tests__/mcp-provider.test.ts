import test from 'ava';
import type { MCPServer } from '@openai/agents-core';
import { ApplicationFailure } from '@temporalio/common';
import { StatelessMCPServerProvider } from '../worker/mcp-provider';

test('StatelessMCPServerProvider refuses a toolMetaResolver without connecting', async (t) => {
  let connected = false;
  const server: MCPServer = {
    cacheToolsList: false,
    name: 'tickets',
    toolMetaResolver: () => ({ tenant: 'tenant-7' }),
    async connect() {
      connected = true;
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
  const activities = new StatelessMCPServerProvider('tickets', () => server)._getActivities();

  for (const name of ['tickets-stateless-list-tools', 'tickets-stateless-call-tool']) {
    const err = await t.throwsAsync(activities[name]!({ toolName: 'echo', args: {} }), {
      instanceOf: ApplicationFailure,
    });
    t.is(err?.type, 'MCPToolMetaResolverUnsupported');
    t.true(err?.nonRetryable);
  }
  t.false(connected);
});
