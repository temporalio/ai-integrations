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

class ProtocolError extends Error {
  constructor(
    readonly code: number,
    message: string,
    readonly data?: unknown
  ) {
    super(message);
    this.name = 'ProtocolError';
  }
}

function failingServer(error: Error): MCPServer {
  return {
    ...fakeServer(undefined),
    async listTools() {
      throw error;
    },
    async callTool() {
      throw error;
    },
  };
}

test('stateless MCP activities fail a JSON-RPC error a retry cannot fix without retries', async (t) => {
  for (const code of [-32700, -32600, -32601, -32602]) {
    const activities = new StatelessMCPServerProvider('files', () =>
      failingServer(new ProtocolError(code, 'rejected', { why: 'x' }))
    )._getActivities();
    for (const name of ['files-stateless-list-tools', 'files-stateless-call-tool']) {
      const err = await t.throwsAsync(activities[name]!({ toolName: 'read', args: {} }), {
        instanceOf: ApplicationFailure,
      });
      t.is(err?.type, 'MCPProtocolError');
      t.true(err?.nonRetryable);
      t.is(err?.message, 'rejected');
      t.deepEqual(err?.details, [code, { why: 'x' }]);
    }
  }
});

test('stateless MCP activities find a protocol error behind a wrapper error', async (t) => {
  const wrapped = new Error('call failed', {
    cause: new ProtocolError(-32602, 'Invalid params'),
  });
  const activities = new StatelessMCPServerProvider('files', () => failingServer(wrapped))._getActivities();
  const err = await t.throwsAsync(activities['files-stateless-call-tool']!({ toolName: 'read', args: {} }), {
    instanceOf: ApplicationFailure,
  });
  t.true(err?.nonRetryable);
});

test('stateless MCP activities keep internal errors and plain errors retryable', async (t) => {
  for (const error of [new ProtocolError(-32603, 'Internal error'), new Error('socket hang up')]) {
    const activities = new StatelessMCPServerProvider('files', () => failingServer(error))._getActivities();
    for (const name of ['files-stateless-list-tools', 'files-stateless-call-tool']) {
      const err = await t.throwsAsync(activities[name]!({ toolName: 'read', args: {} }));
      t.is(err, error);
    }
  }
});
