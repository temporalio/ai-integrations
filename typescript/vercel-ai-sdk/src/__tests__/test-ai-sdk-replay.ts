import { readFileSync } from 'node:fs';
import path from 'node:path';
import { historyFromJSON } from '@temporalio/common/lib/proto-utils';
import { helpers } from '@temporalio/test-helpers';
import { Worker as CoreWorker } from '@temporalio/worker';
import { AiSdkPlugin } from '..';
import { getWeather } from './activities/ai-sdk';
import { makeAiSdkTestFunction } from './helpers/ai-sdk';
import { createTestMcpClient } from './helpers/mcp';
import type { ModelResponse } from './stubs/ai-sdk';
import {
  AbortRecordingProvider,
  helloWorkflowGenerator,
  RecordingProvider,
  TestProvider,
  textResponse,
  toolCallResponse,
  toolsWorkflowGenerator,
} from './stubs/ai-sdk';
import { mcpWorkflow, toolsWorkflow } from './workflows/ai-sdk';

const test = makeAiSdkTestFunction();

test('A history where experimental_download ran an Activity still replays', async (t) => {
  // Recorded before the model Activity downloaded URLs. `experimental_download` got each
  // URL then, and this one ran the `downloadNotes` Activity before the model Activity.
  const histories = path.join(__dirname, '../../src/__tests__/histories');
  const recorded = readFileSync(
    path.join(histories, 'experimental-download-before-url-patch.json'),
    'utf8'
  );
  await CoreWorker.runReplayHistory(
    {
      workflowBundle: t.context.workflowBundle,
      plugins: [new AiSdkPlugin({ modelProvider: new RecordingProvider({}) })],
    },
    historyFromJSON(JSON.parse(recorded))
  );
  t.pass();
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

test('A history where an abort during embedMany cancelled nothing still replays', async (t) => {
  // Recorded before an abort cancelled the model Activity. The signal aborted while
  // `invokeEmbeddingModel` ran, and the Activity then completed.
  const histories = path.join(__dirname, '../../src/__tests__/histories');
  const recorded = readFileSync(
    path.join(histories, 'embed-abort-before-cancel-patch.json'),
    'utf8'
  );
  await CoreWorker.runReplayHistory(
    {
      workflowBundle: t.context.workflowBundle,
      plugins: [new AiSdkPlugin({ modelProvider: new AbortRecordingProvider(false) })],
    },
    historyFromJSON(JSON.parse(recorded))
  );
  t.pass();
});

test('A history with a model call next to a plain Activity still replays', async (t) => {
  // Recorded before a model call accepted an `abortSignal`. Both first Activities
  // completed before one Workflow Task, so the two branches resumed in the same task.
  const histories = path.join(__dirname, '../../src/__tests__/histories');
  const recorded = readFileSync(
    path.join(histories, 'model-next-to-plain-activity.json'),
    'utf8'
  );
  await CoreWorker.runReplayHistory(
    {
      workflowBundle: t.context.workflowBundle,
      plugins: [new AiSdkPlugin({ modelProvider: new TestProvider(helloWorkflowGenerator()) })],
    },
    historyFromJSON(JSON.parse(recorded))
  );
  t.pass();
});
