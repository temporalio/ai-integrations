// A Worker process for the stateful MCP retry test, which kills it to lose the
// session's host. Usage: node stateful-mcp-host.js <address> <taskQueue> <bundlePath>
import type { MCPServer } from '@openai/agents-core';
import { NativeConnection, Worker } from '@temporalio/worker';
import { OpenAIAgentsPlugin, StatefulMCPServerProvider } from '../..';
import { FakeModelProvider } from '../stubs/openai-agents';

/** An MCP server whose stored value lives in this instance, like an MCP session. */
export function memoryServer(name: string, session: string): MCPServer {
  let value = 'EMPTY';
  return {
    name,
    cacheToolsList: false,
    async connect() {},
    async close() {},
    async listTools() {
      return [];
    },
    async callTool(tool, args) {
      if (tool === 'put') value = String(args?.value);
      process.stdout.write(`${tool}\n`);
      return [{ type: 'text', text: `${tool}=${value} session=${session}` }];
    },
    async invalidateToolsCache() {},
  };
}

async function main(): Promise<void> {
  const [address, taskQueue, codePath] = process.argv.slice(2);
  const connection = await NativeConnection.connect({ address });
  const worker = await Worker.create({
    connection,
    taskQueue: taskQueue!,
    workflowBundle: { codePath: codePath! },
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([]),
        mcpServerProviders: [
          new StatefulMCPServerProvider('retried', () => memoryServer('retried', 'A'), connection),
        ],
      }),
    ],
  });
  process.stdout.write('ready\n');
  await worker.run();
}

if (require.main === module) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
