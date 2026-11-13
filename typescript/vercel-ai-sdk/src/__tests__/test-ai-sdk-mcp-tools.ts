import { createMCPClient } from '@ai-sdk/mcp';
import { openai } from '@ai-sdk/openai';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { helpers } from '@temporalio/test-helpers';
import { AiSdkPlugin } from '..';
import { makeAiSdkTestFunction, remoteTests } from './helpers/ai-sdk';
import { createTestMcpClient } from './helpers/mcp';
import type { ModelResponse } from './stubs/ai-sdk';
import { helloWorkflowGenerator, TestProvider, textResponse, toolCallResponse } from './stubs/ai-sdk';
import { mcpSchemaTestWorkflow, mcpWorkflow } from './workflows/ai-sdk';

const test = makeAiSdkTestFunction();

function* mcpGenerator(): Generator<ModelResponse> {
  yield toolCallResponse('list_directory', '/');
  yield textResponse('Some files');
}

test('MCP tool schema survives activity serialization', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new AiSdkPlugin({
        modelProvider: new TestProvider(helloWorkflowGenerator()),
        mcpClientFactories: { testServer: createTestMcpClient },
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(mcpSchemaTestWorkflow);

    // These would FAIL with the bug (undefined), PASS with fix
    t.is(result.schemaType, 'object', 'Schema type should be "object", not undefined');
    t.true(result.hasProperties, 'Schema should have properties');
    t.is(result.propertyDescription, 'Test parameter', 'Property description should be preserved');
  });
});

// Currently fails in CI due to invalid server response but passes locally
test.skip('MCP Use', async (t) => {
  if (remoteTests) {
    t.timeout(120 * 1000);
  }
  const { createWorker, executeWorkflow } = helpers(t);

  const mcpClientFactories = {
    testServer: () =>
      createMCPClient({
        transport: new StdioClientTransport({
          command: 'npx',
          args: ['-y', '@modelcontextprotocol/server-filesystem@latest', __dirname],
        }),
      }),
  };
  const worker = await createWorker({
    plugins: [
      new AiSdkPlugin({
        modelProvider: remoteTests ? openai : new TestProvider(mcpGenerator()),
        mcpClientFactories,
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(mcpWorkflow, {
      args: ['Tell me what files you know about. Use your tools.'],
    });

    t.assert(result);
    if (!remoteTests) {
      t.is('Some files', result);
    }
  });
});
