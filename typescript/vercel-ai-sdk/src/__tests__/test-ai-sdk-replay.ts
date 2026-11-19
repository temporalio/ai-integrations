import path from 'node:path';
import { helpers, loadHistory } from '@temporalio/test-helpers';
import { Worker as CoreWorker } from '@temporalio/worker';
import { AiSdkPlugin } from '..';
import { getWeather } from './activities/ai-sdk';
import { makeAiSdkTestFunction } from './helpers/ai-sdk';
import { createTestMcpClient } from './helpers/mcp';
import type { ModelResponse } from './stubs/ai-sdk';
import {
  helloWorkflowGenerator,
  TestProvider,
  textResponse,
  toolCallResponse,
  toolsWorkflowGenerator,
} from './stubs/ai-sdk';
import { mcpWorkflow, toolsWorkflow } from './workflows/ai-sdk';

const test = makeAiSdkTestFunction();

// Replay is promised only within a major (see AGENTS.md), so only the histories
// in the current major's folder replay.
const REPLAY_MAJOR = 2;
const histories = path.resolve(__dirname, '../../src/__tests__/histories', `${REPLAY_MAJOR}.x`);

test('A history whose reasoning metadata the Workflow never saw still replays', async (t) => {
  // Recorded with `reasoningMetadataTimerWorkflow` on main at 624a3d0. The model sent
  // provider metadata on `reasoning-end`, so the Activity result holds it, but the Workflow
  // replayed the block with none and started no timer.
  const history = await loadHistory(path.join(histories, 'reasoning-metadata-on-end-624a3d0.json'));
  await t.notThrowsAsync(
    CoreWorker.runReplayHistory(
      {
        workflowBundle: t.context.workflowBundle,
        plugins: [new AiSdkPlugin({ modelProvider: new TestProvider(helloWorkflowGenerator()) })],
      },
      history
    )
  );
});

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
