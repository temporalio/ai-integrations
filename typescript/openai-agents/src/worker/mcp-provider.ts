import type { MCPServer } from '@openai/agents-core';
import { ApplicationFailure } from '@temporalio/common';
import { MCP_CALL_TOOL_SUFFIX, MCP_LIST_TOOLS_SUFFIX, MCP_STATELESS_SUFFIX } from '../common/mcp-types';

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
      type: 'MCPToolFilterUnsupported',
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

export class StatelessMCPServerProvider {
  private readonly _name: string;

  constructor(
    public readonly name: string,
    private serverFactory: (factoryArgument?: unknown) => MCPServer
  ) {
    this._name = `${name}${MCP_STATELESS_SUFFIX}`;
  }

  _getActivities(): Record<string, (...args: any[]) => Promise<unknown>> {
    return {
      [`${this._name}${MCP_LIST_TOOLS_SUFFIX}`]: async (input?: { factoryArgument?: unknown }) => {
        const s = this.serverFactory(input?.factoryArgument);
        rejectCallableToolFilter(s);
        try {
          await s.connect();
          return filterListedTools(s, await s.listTools());
        } finally {
          await s.close();
        }
      },
      [`${this._name}${MCP_CALL_TOOL_SUFFIX}`]: async (input: {
        toolName: string;
        args: Record<string, unknown> | null;
        factoryArgument?: unknown;
      }) => {
        const s = this.serverFactory(input.factoryArgument);
        rejectCallableToolFilter(s);
        try {
          await s.connect();
          return await s.callTool(input.toolName, input.args);
        } finally {
          await s.close();
        }
      },
    };
  }
}
