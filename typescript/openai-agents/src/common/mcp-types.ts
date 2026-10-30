import type { MCPServer, MCPToolMetaResolver } from '@openai/agents-core';

export const MCP_LIST_TOOLS_SUFFIX = '-list-tools';
export const MCP_CALL_TOOL_SUFFIX = '-call-tool';
export const MCP_SERVER_SESSION_SUFFIX = '-server-session';
export const MCP_STATEFUL_SUFFIX = '-stateful';
export const MCP_STATELESS_SUFFIX = '-stateless';

/** Error type for dedicated-Worker scheduling and heartbeat failures. */
export const DEDICATED_WORKER_FAILURE_TYPE = 'DedicatedWorkerFailure';

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
