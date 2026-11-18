import test from 'ava';
import type { MCPServer } from '@openai/agents-core';
import { ApplicationFailure } from '@temporalio/common';
import { TOOL_META_RESOLVER_UNSUPPORTED_TYPE } from '..';
import { TOOL_FILTER_UNSUPPORTED_TYPE } from '../common/mcp-types';
import { StatelessMCPServerProvider, filterListedTools } from '../worker/mcp-provider';
import { StatefulMCPServerProvider } from '../worker/stateful-mcp-provider';
import { statelessMcpServer } from '../workflow/mcp-client';

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

test('StatelessMCPServerProvider rejects invalid factory results before connecting or closing', async (t) => {
  let connects = 0;
  let closes = 0;
  const incompleteServer = {
    ...fakeServer(undefined, () => connects++),
    async close() {
      closes++;
    },
    callTool: undefined,
  };
  const invalidResults: [unknown, string][] = [
    [undefined, 'undefined'],
    [null, 'null'],
    [42, 'number'],
    [{}, 'Object'],
    [Promise.resolve(fakeServer(undefined)), 'Promise'],
    [incompleteServer, 'Object'],
  ];
  for (const [value, kind] of invalidResults) {
    const activities = new StatelessMCPServerProvider('files', () => value as MCPServer)._getActivities();
    for (const name of ['files-stateless-list-tools', 'files-stateless-call-tool']) {
      const err = await t.throwsAsync(activities[name]!({ toolName: 'read', args: {} }), {
        instanceOf: ApplicationFailure,
      });
      t.is(err?.type, 'MCPServerFactoryTypeError');
      t.true(err?.nonRetryable);
      t.is(
        err?.message,
        `MCP server factory 'files' returned ${kind}, ` +
          'not a server with connect(), close(), listTools() and callTool() methods.'
      );
    }
  }
  t.is(connects, 0);
  t.is(closes, 0);
});

test('a rejecting async factory fails the Activity without an unhandled rejection', async (t) => {
  const unhandled: unknown[] = [];
  const listener = (reason: unknown) => unhandled.push(reason);
  process.on('unhandledRejection', listener);
  try {
    const activities = new StatelessMCPServerProvider('files', (async () => {
      throw new Error('server not ready');
    }) as unknown as () => MCPServer)._getActivities();
    const err = await t.throwsAsync(activities['files-stateless-list-tools']!({}), { instanceOf: ApplicationFailure });
    t.is(err?.type, 'MCPServerFactoryTypeError');
    // Unhandled rejections surface after the microtask queue drains and a macrotask turn.
    await new Promise((resolve) => setTimeout(resolve, 20));
    t.deepEqual(unhandled, []);
  } finally {
    process.off('unhandledRejection', listener);
  }
});

test('a thenable factory result is rejected without calling its then', async (t) => {
  let thenCalls = 0;
  const lazy = {
    then() {
      thenCalls++;
    },
  };
  const throwing = {
    get then(): never {
      throw new Error('then getter');
    },
  };
  for (const result of [lazy, throwing]) {
    const activities = new StatelessMCPServerProvider('files', (() => result) as unknown as () => MCPServer)._getActivities();
    const err = await t.throwsAsync(activities['files-stateless-list-tools']!({}), { instanceOf: ApplicationFailure });
    t.is(err?.type, 'MCPServerFactoryTypeError');
    t.true(err?.nonRetryable);
  }
  t.is(thenCalls, 0);
});

test('MCP server providers refuse a factory that is not a function when they are built', (t) => {
  const invalidFactories: [unknown, string][] = [
    [undefined, 'undefined'],
    [null, 'null'],
    [42, 'number'],
    [fakeServer(undefined), 'Object'],
  ];
  for (const [value, kind] of invalidFactories) {
    const factory = value as () => MCPServer;
    for (const build of [
      () => new StatelessMCPServerProvider('files', factory),
      () => new StatefulMCPServerProvider('files', factory, undefined as never),
    ]) {
      t.throws(build, { instanceOf: TypeError, message: `MCP server factory 'files' is ${kind}, not a function.` });
    }
  }
});

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

test('statelessMcpServer keeps cacheToolsList assignable and reports false with a filter', (t) => {
  const server = statelessMcpServer('files');
  t.true(server.cacheToolsList);
  server.cacheToolsList = false;
  t.false(server.cacheToolsList);

  const filtered = statelessMcpServer('files', { toolFilter: { blockedToolNames: ['delete'] } });
  t.false(filtered.cacheToolsList);
  filtered.cacheToolsList = true;
  t.false(filtered.cacheToolsList);
});

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
    t.is(err?.type, TOOL_META_RESOLVER_UNSUPPORTED_TYPE);
    t.true(err?.nonRetryable);
  }
  t.is(connects, 0);
});
