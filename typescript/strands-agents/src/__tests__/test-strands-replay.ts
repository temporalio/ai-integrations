import path from 'node:path';
import type { TestFn } from 'ava';
import type { BaseContext } from '@temporalio/test-helpers';
import { test as anyTest, helpers, loadHistory } from '@temporalio/test-helpers';
import { Worker } from '@temporalio/worker';
import { DeterminismViolationError } from '@temporalio/workflow';
import { StrandsPlugin } from '..';
import { getWeather } from './activities/strands';
import { setupTestEnv, StubModel, textTurn, toolCallTurn } from './helpers/strands';
import { activityToolAgent } from './workflows/strands';

const test = anyTest as TestFn<BaseContext>;

setupTestEnv(test);

test('a history recorded before the model config read fails to replay', async (t) => {
  // Recorded with `helloAgent` on 73deca0, before agents read the model config: its first command
  // is `invokeModel`, where an agent now schedules `strandsGetModelConfig`.
  const history = await loadHistory(
    path.resolve(__dirname, '../../src/__tests__/histories/hello-agent-before-model-config.json')
  );
  const err = await t.throwsAsync(
    Worker.runReplayHistory(
      {
        workflowBundle: t.context.workflowBundle,
        plugins: [new StrandsPlugin({ models: { test: () => new StubModel([]) } })],
      },
      history
    ),
    { instanceOf: DeterminismViolationError }
  );
  t.regex(err!.message, /\[TMPRL1100\].*'invokeModel'.*'strandsGetModelConfig'/);
});

test('a history whose code depends on when a handler finishes still replays', async (t) => {
  // Recorded with `handlerTiming` on main at 63d7aec: one microtask after its Signal handler ran,
  // every handler had finished, so the run scheduled `finished`. The plugin wraps no handlers, so
  // they finish at the same moment and the history replays.
  const history = await loadHistory(
    path.resolve(__dirname, '../../src/__tests__/histories/handler-timing-63d7aec.json')
  );
  await t.notThrowsAsync(
    Worker.runReplayHistory(
      {
        workflowBundle: t.context.workflowBundle,
        plugins: [new StrandsPlugin({ models: { test: () => new StubModel([]) } })],
      },
      history
    )
  );
});

test('histories of a re-listing beside another coroutine recorded on main still replay', async (t) => {
  // Recorded on main at 017d88a, where each `cacheTools: false` client had its own `AfterToolsEvent`
  // hook. The re-listing of `b` and another coroutine's activity completed in one workflow task, so
  // whether `a-listTools` or that coroutine's next activity comes next depends on how many
  // microtasks each takes to schedule it. Main scheduled `second`'s `c-listTools` first in
  // `mcpAgentsSideBySide`, and `a-listTools` first in `mcpAgentBesideCoroutine`.
  for (const file of ['mcp-side-by-side-017d88a.json', 'mcp-beside-coroutine-017d88a.json']) {
    const history = await loadHistory(path.resolve(__dirname, '../../src/__tests__/histories', file));
    await t.notThrowsAsync(
      Worker.runReplayHistory(
        {
          workflowBundle: t.context.workflowBundle,
          plugins: [new StrandsPlugin({ models: { first: () => new StubModel([]), second: () => new StubModel([]) } })],
        },
        history
      ),
      file
    );
  }
});

test('captured history replays deterministically', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const plugin = (): StrandsPlugin =>
    new StrandsPlugin({
      models: {
        test: () =>
          new StubModel([
            toolCallTurn('getWeather', 'call_1', { location: 'Kyoto' }),
            textTurn('the weather is sunny'),
          ]),
      },
    });

  const worker = await createWorker({ plugins: [plugin()], activities: { getWeather } });

  let history;
  await worker.runUntil(async () => {
    const handle = await startWorkflow(activityToolAgent, {
      args: ['weather in Tokyo?'],
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(await handle.result(), 'the weather is sunny');
    history = await handle.fetchHistory();
  });

  // Replay the captured history with a fresh plugin instance. Any non-determinism
  // (e.g. a polyfill that read system state) would surface as DeterminismViolationError.
  await Worker.runReplayHistory(
    {
      workflowBundle: t.context.workflowBundle,
      plugins: [plugin()],
    },
    history!
  );
});
