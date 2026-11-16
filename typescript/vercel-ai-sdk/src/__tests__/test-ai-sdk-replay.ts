import { helpers } from '@temporalio/test-helpers';
import { Worker as CoreWorker } from '@temporalio/worker';
import { AiSdkPlugin } from '..';
import { getWeather } from './activities/ai-sdk';
import { makeAiSdkTestFunction } from './helpers/ai-sdk';
import { createTestMcpClient } from './helpers/mcp';
import type { ModelResponse } from './stubs/ai-sdk';
import { TestProvider, textResponse, toolCallResponse, toolsWorkflowGenerator } from './stubs/ai-sdk';
import { mcpWorkflow, toolsWorkflow } from './workflows/ai-sdk';

const test = makeAiSdkTestFunction();

test('Captured multi-step tools history replays deterministically', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const plugin = () => new AiSdkPlugin({ modelProvider: new TestProvider(toolsWorkflowGenerator()) });

  const worker = await createWorker({ plugins: [plugin()], activities: { getWeather } });

  let history;
  await worker.runUntil(async () => {
    const handle = await startWorkflow(toolsWorkflow, {
      args: ['What is the weather in Tokyo?'],
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(await handle.result(), 'Test weather result');
    history = await handle.fetchHistory();
  });

  // Replay the captured history with a fresh plugin instance. Any non-determinism
  // in the ai@7 agent loop or the sandbox polyfills would surface as
  // DeterminismViolationError.
  await CoreWorker.runReplayHistory({ workflowBundle: t.context.workflowBundle, plugins: [plugin()] }, history!);
  t.pass();
});

function* mcpReplayGenerator(): Generator<ModelResponse> {
  yield toolCallResponse('testTool', '{"testParam":"hello"}');
  yield textResponse('Done');
}

test('Captured multi-step MCP history replays deterministically', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const plugin = () =>
    new AiSdkPlugin({
      modelProvider: new TestProvider(mcpReplayGenerator()),
      mcpClientFactories: { testServer: createTestMcpClient },
    });

  const worker = await createWorker({ plugins: [plugin()] });

  let history;
  await worker.runUntil(async () => {
    const handle = await startWorkflow(mcpWorkflow, {
      args: ['What files do you have? Use your tools.'],
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(await handle.result(), 'Done');
    history = await handle.fetchHistory();
  });

  await CoreWorker.runReplayHistory({ workflowBundle: t.context.workflowBundle, plugins: [plugin()] }, history!);
  t.pass();
});
