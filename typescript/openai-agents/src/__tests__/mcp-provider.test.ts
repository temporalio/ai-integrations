import test from 'ava';
import type { MCPServer } from '@openai/agents-core';
import { ApplicationFailure } from '@temporalio/common';
import { TOOL_FILTER_UNSUPPORTED_TYPE } from '../common/mcp-types';
import { StatelessMCPServerProvider, filterListedTools } from '../worker/mcp-provider';

function fakeServer(toolFilter: MCPServer['toolFilter'], onConnect = () => {}): MCPServer {
  return {
    cacheToolsList: false,
    name: 'files',
    toolFilter,
    async connect() {
      onConnect();
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

test('filterListedTools applies a static filter with the agent SDK rules', (t) => {
  const tools = ['read', 'read_all', 'delete'].map((name) => ({ name }));
  const names = (toolFilter: MCPServer['toolFilter']) =>
    filterListedTools(fakeServer(toolFilter), tools).map((tool) => tool.name);

  t.deepEqual(names(undefined), ['read', 'read_all', 'delete']);
  t.deepEqual(names({ allowedToolNames: [], blockedToolNames: [] }), ['read', 'read_all', 'delete']);
  // Names match whole, not as prefixes.
  t.deepEqual(names({ allowedToolNames: ['read'] }), ['read']);
  t.deepEqual(names({ blockedToolNames: ['read'] }), ['read_all', 'delete']);
  // A blocked name stays blocked when it is also allowed.
  t.deepEqual(names({ allowedToolNames: ['read', 'delete'], blockedToolNames: ['delete'] }), ['read']);
});

test('StatelessMCPServerProvider refuses a callable toolFilter without connecting', async (t) => {
  let connected = false;
  const provider = new StatelessMCPServerProvider('files', () =>
    fakeServer(
      async () => true,
      () => {
        connected = true;
      }
    )
  );
  const activities = provider._getActivities();

  for (const name of ['files-stateless-list-tools', 'files-stateless-call-tool']) {
    const err = await t.throwsAsync(activities[name]!({ toolName: 'read', args: {} }), {
      instanceOf: ApplicationFailure,
    });
    t.is(err?.type, TOOL_FILTER_UNSUPPORTED_TYPE);
    t.true(err?.nonRetryable);
  }
  t.false(connected);
});
