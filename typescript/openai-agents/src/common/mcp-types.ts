import type { MCPServer, MCPToolFilterCallable, MCPToolFilterStatic, MCPToolMetaResolver } from '@openai/agents-core';

export const MCP_LIST_TOOLS_SUFFIX = '-list-tools';
export const MCP_CALL_TOOL_SUFFIX = '-call-tool';
export const MCP_SERVER_SESSION_SUFFIX = '-server-session';
export const MCP_STATEFUL_SUFFIX = '-stateful';
export const MCP_STATELESS_SUFFIX = '-stateless';

/** Error type for dedicated-Worker scheduling and heartbeat failures. */
export const DEDICATED_WORKER_FAILURE_TYPE = 'DedicatedWorkerFailure';

/** Error type for a callable `toolFilter` on the server the factory returns, which the Worker refuses. */
export const TOOL_FILTER_UNSUPPORTED_TYPE = 'MCPToolFilterUnsupported';

/** Error type for a `toolMetaResolver` on the server the factory returns, which the Worker refuses. */
export const TOOL_META_RESOLVER_UNSUPPORTED_TYPE = 'MCPToolMetaResolverUnsupported';

/**
 * Stateful variant of `MCPServer`. Adds `cleanup()` — the documented
 * lifecycle method users invoke from `finally` blocks after `connect()`.
 */
export interface StatefulTemporalMCPServer extends MCPServer {
  cleanup(): Promise<void>;
}

/** Options both MCP handles take. The agent reads them from the handle in the Workflow, as from a native MCP server. */
export interface McpServerHandleOptions {
  /**
   * Tool filter the agent applies in the Workflow each time it lists tools, as for a native MCP
   * server. A callable filter gets the agent, the run context and the name the server was
   * registered with, and must be deterministic.
   */
  toolFilter?: MCPToolFilterCallable | MCPToolFilterStatic;
  /**
   * Builds each tool call's MCP `_meta` in the Workflow, as for a native MCP server. It gets the
   * run context and must be deterministic. Older plugin versions ignore this option and may not
   * replay a History in which it ran, because running it can change the Workflow's commands or
   * their order.
   */
  toolMetaResolver?: MCPToolMetaResolver;
}

// Forwarded per-call (not stored on the provider) so different plugins sharing
// a provider each get their own per-run config.
export interface StatefulMcpActivityInterceptorOptions {
  addTemporalSpans?: boolean;
}

// Input shape of the `${name}-stateful-server-session` Activity.
export interface StatefulMcpServerSessionArgs {
  factoryArgument?: unknown;
  activityInterceptorOptions?: StatefulMcpActivityInterceptorOptions;
}
