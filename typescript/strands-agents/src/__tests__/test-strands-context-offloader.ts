import type { Message, ModelStreamEvent, StreamOptions, TextBlock } from '@strands-agents/sdk';
import { ToolResultBlock } from '@strands-agents/sdk';
import type { TestFn } from 'ava';
import type { BaseContext } from '@temporalio/test-helpers';
import { test as anyTest, helpers } from '@temporalio/test-helpers';
import { Worker } from '@temporalio/worker';
import { StrandsPlugin } from '..';
import { setupTestEnv, StubModel, textTurn, toolCallTurn } from './helpers/strands';
import { contextOffloaderAgent } from './workflows/strands';

const test = anyTest as TestFn<BaseContext>;

setupTestEnv(test);

const DOCUMENT = 'A large document with Unicode content: 日本語 🌍.\n'.repeat(200);

/** Read the reference from the actual preview, then ask Strands' retrieval tool for its content. */
class OffloaderModel extends StubModel {
  private call = 0;

  constructor(private readonly results: ToolResultBlock[][]) {
    super([]);
  }

  override async *stream(messages: Message[], _options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
    const results = messages.flatMap((message) =>
      message.content.filter((block): block is ToolResultBlock => block instanceof ToolResultBlock)
    );
    this.results.push(results);
    switch (++this.call) {
      case 1:
        yield* toolCallTurn('loadDocument', 'load_1', {});
        break;
      case 2: {
        const preview = (results[0]?.content[0] as TextBlock | undefined)?.text ?? '';
        const reference = /^\s+(\S+) \(text,/m.exec(preview)?.[1];
        if (!reference) throw new Error(`No offloaded reference in tool result: ${preview}`);
        yield* toolCallTurn('retrieve_offloaded_content', 'retrieve_1', { reference });
        break;
      }
      case 3:
        yield* textTurn('retrieved');
        break;
      case 4:
        yield* textTurn('next');
        break;
      default:
        throw new Error('Unexpected model call');
    }
  }
}

for (const [storageKind, evictAfterCycles] of [
  ['memory', null],
  ['memory', 1],
  ['default', null],
] as const) {
  test(`native context offloader stores and retrieves with ${storageKind} storage, eviction ${evictAfterCycles}`, async (t) => {
    const { createWorker, startWorkflow } = helpers(t);
    const results: ToolResultBlock[][] = [];
    let loads = 0;
    const plugin = (): StrandsPlugin => new StrandsPlugin({ models: { test: () => new OffloaderModel(results) } });
    const worker = await createWorker({
      plugins: [plugin()],
      // Every model call starts in a fresh Workflow instance, rebuilding storage from history.
      maxCachedWorkflows: 0,
      activities: {
        loadDocument: async () => {
          loads++;
          return DOCUMENT;
        },
      },
    });

    const history = await worker.runUntil(async () => {
      const handle = await startWorkflow(contextOffloaderAgent, {
        args: [storageKind, evictAfterCycles],
        workflowExecutionTimeout: '30 seconds',
      });
      const result = await handle.result();
      t.deepEqual(result.replies, ['retrieved', 'next']);
      t.is(result.error, undefined);
      if (storageKind === 'memory') {
        t.deepEqual(result.keys, evictAfterCycles === null ? ['offloader/load_1_0'] : []);
      }
      return handle.fetchHistory();
    });

    t.is(loads, 1);
    t.is(results.length, 4);
    const preview = (results[1]?.[0]?.content[0] as TextBlock).text;
    t.true(preview.startsWith('[Offloaded:'));
    t.true(preview.length < DOCUMENT.length);
    t.false(preview.includes(DOCUMENT));
    const retrieved = results[2]?.find((result) => result.toolUseId === 'retrieve_1');
    t.is(retrieved?.status, 'success');
    t.is((retrieved?.content[0] as TextBlock).text, DOCUMENT);

    await t.notThrowsAsync(
      Worker.runReplayHistory({ workflowBundle: t.context.workflowBundle, plugins: [plugin()] }, history)
    );
    t.is(loads, 1);
    t.is(results.length, 4);
  });
}

test('legacy offloader InMemoryStorage fails at initialization on the Workflow WeakRef guard', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);
  const worker = await createWorker({
    plugins: [new StrandsPlugin({ models: { test: () => new StubModel([]) } })],
  });
  const result = await worker.runUntil(
    executeWorkflow(contextOffloaderAgent, {
      args: ['legacy', null],
      workflowExecutionTimeout: '30 seconds',
    })
  );
  t.deepEqual(result.replies, []);
  t.is(result.error?.name, 'DeterminismViolationError');
  t.regex(result.error?.message ?? '', /WeakRef cannot be used in Workflows/);
});
