import type { MCPServer } from '@openai/agents-core';
import { log } from '@temporalio/activity';
import { MCP_CALL_TOOL_SUFFIX, MCP_LIST_TOOLS_SUFFIX, MCP_STATELESS_SUFFIX } from '../common/mcp-types';

/**
 * Calls a tool with the `_meta` the Workflow built, as the agent does for a native server. A
 * `toolMetaResolver` on the Worker-side server needs the run context, which only the Workflow
 * has, so it is skipped with a warning.
 */
export async function callToolWithMeta(
  server: MCPServer,
  toolName: string,
  args: Record<string, unknown> | null,
  meta: Record<string, unknown> | undefined
): ReturnType<MCPServer['callTool']> {
  if (server.toolMetaResolver) {
    log.warn(
      `MCP server '${server.name}' has a toolMetaResolver, which the Worker cannot run. ` +
        'Pass it as the toolMetaResolver option of statelessMcpServer() or statefulMcpServer() in the Workflow.'
    );
  }
  return meta === undefined ? server.callTool(toolName, args) : server.callTool(toolName, args, meta);
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
        try {
          await s.connect();
          return await s.listTools();
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
        try {
          await s.connect();
          return await callToolWithMeta(s, input.toolName, input.args, input.meta);
        } finally {
          await s.close();
        }
      },
    };
  }
}
