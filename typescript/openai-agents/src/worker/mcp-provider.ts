import type { MCPServer } from '@openai/agents-core';
import { ApplicationFailure } from '@temporalio/common';
import {
  MCP_CALL_TOOL_SUFFIX,
  MCP_LIST_TOOLS_SUFFIX,
  MCP_PROTOCOL_ERROR_TYPE,
  MCP_STATELESS_SUFFIX,
  TOOL_FILTER_UNSUPPORTED_TYPE,
  TOOL_META_RESOLVER_UNSUPPORTED_TYPE,
} from '../common/mcp-types';

// Names a value for an error message: `undefined`, `null`, `number`, or an object's class, such as `MCPServerStdio`.
function describeValue(value: unknown): string {
  if (value === null) return 'null';
  if (typeof value !== 'object' && typeof value !== 'function') return typeof value;
  return (value as { constructor?: { name?: string } }).constructor?.name || typeof value;
}

/** Refuse a factory that isn't a function when the provider is built, not in every MCP Activity. */
export function assertMCPServerFactory(name: string, factory: unknown): void {
  if (typeof factory !== 'function') {
    throw new TypeError(`MCP server factory '${name}' is ${describeValue(factory)}, not a function.`);
  }
}

/** Check that a factory result has the methods the Worker calls. */
export function assertMCPServer(name: string, server: unknown): asserts server is MCPServer {
  const methods = ['connect', 'close', 'listTools', 'callTool'] as const;
  if (server != null && methods.every((method) => typeof (server as MCPServer)[method] === 'function')) return;
  throw ApplicationFailure.create({
    message:
      `MCP server factory '${name}' returned ${describeValue(server)}, ` +
      'not a server with connect(), close(), listTools() and callTool() methods.',
    type: 'MCPServerFactoryTypeError',
    nonRetryable: true,
  });
}

/**
 * Fails on a callable `toolFilter` on the server the factory returns. It needs the agent and the
 * run context, which exist only in the Workflow, so the Worker cannot apply it.
 */
export function rejectCallableToolFilter(server: MCPServer): void {
  if (typeof server.toolFilter === 'function') {
    throw ApplicationFailure.create({
      message:
        `MCP server '${server.name}' has a callable toolFilter, which the Worker cannot run because the agent ` +
        'and run context it gets exist only in the Workflow. Pass it as the toolFilter option of ' +
        'statelessMcpServer() or statefulMcpServer() instead.',
      type: TOOL_FILTER_UNSUPPORTED_TYPE,
      nonRetryable: true,
    });
  }
}

/**
 * Applies the static `toolFilter` of the server the factory returns, as the agent does for a native
 * server. A callable filter is refused by `rejectCallableToolFilter()` before this runs.
 */
export function filterListedTools<T extends { name: string }>(server: MCPServer, tools: T[]): T[] {
  const filter = server.toolFilter;
  if (!filter || typeof filter === 'function') {
    return tools;
  }
  const allowed = filter.allowedToolNames ?? [];
  const blocked = filter.blockedToolNames ?? [];
  return tools.filter((tool) => (allowed.length === 0 || allowed.includes(tool.name)) && !blocked.includes(tool.name));
}

/**
 * Fails on a `toolMetaResolver` on the server the factory returns. It needs the run context, which
 * exists only in the Workflow, so the Worker cannot run it.
 */
export function rejectToolMetaResolver(server: MCPServer): void {
  if (server.toolMetaResolver) {
    throw ApplicationFailure.create({
      message:
        `MCP server '${server.name}' has a toolMetaResolver, which the Worker cannot run because the run ` +
        'context it gets exists only in the Workflow. Pass it as the toolMetaResolver option of ' +
        'statelessMcpServer() or statefulMcpServer() instead.',
      type: TOOL_META_RESOLVER_UNSUPPORTED_TYPE,
      nonRetryable: true,
    });
  }
}

/** Calls a tool with the `_meta` the Workflow built, as the agent does for a native server. */
export async function callToolWithMeta(
  server: MCPServer,
  toolName: string,
  args: Record<string, unknown> | null,
  meta: Record<string, unknown> | undefined
): ReturnType<MCPServer['callTool']> {
  return meta === undefined ? server.callTool(toolName, args) : server.callTool(toolName, args, meta);
}

// JSON-RPC errors that a retry of the same request cannot fix: parse error, invalid request,
// method not found and invalid params. INTERNAL_ERROR, transport errors, timeouts and unknown
// codes stay retryable.
const NON_RETRYABLE_PROTOCOL_ERROR_CODES = new Set([-32700, -32600, -32601, -32602]);

// The MCP SDK raises `McpError` (v1) or `ProtocolError` (v2) for a JSON-RPC error response.
function protocolErrorOf(err: unknown): { code: number; message: string; data?: unknown } | undefined {
  for (let current = err, depth = 0; current instanceof Error && depth < 8; current = current.cause, depth++) {
    const { name, code } = current as Error & { code?: unknown };
    if ((name === 'ProtocolError' || name === 'McpError') && typeof code === 'number') {
      return current as Error & { code: number; data?: unknown };
    }
  }
  return undefined;
}

/**
 * Runs an MCP request. A JSON-RPC error a retry cannot fix (invalid params, invalid request,
 * method not found, parse error) becomes a non-retryable `MCPProtocolError` failure instead of
 * a plain error that the Activity retry policy would retry until it gives up. Any other error
 * is rethrown unchanged.
 */
export async function failNonRetryableProtocolError<T>(request: () => Promise<T>): Promise<T> {
  try {
    return await request();
  } catch (err) {
    const protocolError = protocolErrorOf(err);
    if (protocolError && NON_RETRYABLE_PROTOCOL_ERROR_CODES.has(protocolError.code)) {
      throw ApplicationFailure.create({
        message: protocolError.message,
        type: MCP_PROTOCOL_ERROR_TYPE,
        nonRetryable: true,
        details: protocolError.data === undefined ? [protocolError.code] : [protocolError.code, protocolError.data],
        cause: err instanceof Error ? err : undefined,
      });
    }
    throw err;
  }
}

export class StatelessMCPServerProvider {
  private readonly _name: string;

  constructor(
    public readonly name: string,
    private serverFactory: (factoryArgument?: unknown) => MCPServer
  ) {
    assertMCPServerFactory(name, serverFactory);
    this._name = `${name}${MCP_STATELESS_SUFFIX}`;
  }

  _getActivities(): Record<string, (...args: any[]) => Promise<unknown>> {
    return {
      [`${this._name}${MCP_LIST_TOOLS_SUFFIX}`]: async (input?: { factoryArgument?: unknown }) => {
        const s = this.serverFactory(input?.factoryArgument);
        assertMCPServer(this.name, s);
        rejectCallableToolFilter(s);
        rejectToolMetaResolver(s);
        try {
          await s.connect();
          return filterListedTools(s, await failNonRetryableProtocolError(() => s.listTools()));
        } finally {
          await s.close();
        }
      },
      [`${this._name}${MCP_CALL_TOOL_SUFFIX}`]: async (input: {
        toolName: string;
        args: Record<string, unknown> | null;
        meta?: Record<string, unknown>;
        factoryArgument?: unknown;
      }) => {
        const s = this.serverFactory(input.factoryArgument);
        assertMCPServer(this.name, s);
        rejectCallableToolFilter(s);
        rejectToolMetaResolver(s);
        try {
          await s.connect();
          return await failNonRetryableProtocolError(() => callToolWithMeta(s, input.toolName, input.args, input.meta));
        } finally {
          await s.close();
        }
      },
    };
  }
}
