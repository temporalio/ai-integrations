// Stub module that the StrandsPlugin's bundler config uses to replace
// worker-only transport modules and `node:*` schemes that
// `@strands-agents/sdk/mcp.js`/`mcp-config.js` import statically but that
// the workflow never reaches at runtime.
//
// `@strands-agents/sdk/mcp.js` does named imports like
// `import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js'`,
// so a pure-empty module would fail webpack's ES-module named-export check
// at build time. Export every name MCP/Strands' static imports reach as a
// no-op class. Workflow code never instantiates these — the real transports
// run worker-side from the user's `mcpClients` factories.
// MCP v2 bundles its HTTP transport into the client entrypoint and statically
// imports EventSourceParserStream from the rewritten eventsource-parser/stream
// module, so that name must also remain available to the bundler.

class WorkflowSandboxStub {}

export {
  WorkflowSandboxStub as StreamableHTTPClientTransport,
  WorkflowSandboxStub as StdioClientTransport,
  WorkflowSandboxStub as SSEClientTransport,
  WorkflowSandboxStub as EventSource,
  WorkflowSandboxStub as EventSourceParserStream,
};
export default WorkflowSandboxStub;
