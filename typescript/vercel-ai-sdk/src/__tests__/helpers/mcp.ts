import type { MCPClient } from '@ai-sdk/mcp';
import { createMCPClient } from '@ai-sdk/mcp';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';

// Create in-memory MCP server with test tool
export const createTestMcpClient = async (): Promise<MCPClient> => {
  // eslint-disable-next-line @typescript-eslint/no-deprecated
  const server = new Server({ name: 'test-server', version: '1.0.0' }, { capabilities: { tools: {} } });

  // Register tools/list handler
  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [
      {
        name: 'testTool',
        description: 'A test tool',
        inputSchema: {
          type: 'object',
          properties: {
            testParam: { type: 'string', description: 'Test parameter' },
          },
          required: ['testParam'],
        },
      },
    ],
  }));

  // Register tools/call handler
  server.setRequestHandler(CallToolRequestSchema, async () => ({
    content: [{ type: 'text', text: 'ok' }],
  }));

  // Create linked in-memory transports
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await server.connect(serverTransport);

  // Return real AI SDK MCP client connected to in-memory server
  return createMCPClient({ transport: clientTransport });
};
