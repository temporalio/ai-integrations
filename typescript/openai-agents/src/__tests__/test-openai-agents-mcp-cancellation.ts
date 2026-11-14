import { Context } from '@temporalio/activity';
import { WorkflowFailedError } from '@temporalio/client';
import { CancelledFailure } from '@temporalio/common';
import { temporal } from '@temporalio/proto';
import { helpers } from '@temporalio/test-helpers';
import { OpenAIAgentsPlugin, StatefulMCPServerProvider } from '..';
import { memoryServer } from './helpers/stateful-mcp-host';
import {
  statefulMcpSessionEndedWorkflow,
  statefulMcpCancelledCallerWorkflow,
  statefulMcpCancelledAfterRefusalWorkflow,
  resumeSignal,
} from './workflows/openai-agents';
import { FakeModelProvider } from './stubs/openai-agents';
import { makeOpenAIAgentsTestFunction } from './helpers/openai-agents';
import EventType = temporal.api.enums.v1.EventType;

const test = makeOpenAIAgentsTestFunction();

test('Stateful MCP: a Workflow cancellation wins over the session it ended', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const calls: string[] = [];
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([]),
        mcpServerProviders: [
          new StatefulMCPServerProvider(
            'sessionEnded',
            () => {
              const server = memoryServer('sessionEnded', 'S', calls);
              server.callTool = async () => {
                calls.push('hold');
                await Context.current().sleep(30_000).catch(() => {});
                return [];
              };
              return server;
            },
            t.context.env.nativeConnection
          ),
        ],
      }),
    ],
  });
  await worker.runUntil(async () => {
    const handle = await startWorkflow(statefulMcpCancelledCallerWorkflow, { workflowExecutionTimeout: '30 seconds' });
    for (let i = 0; i < 300 && !calls.includes('hold'); i++) await new Promise((r) => setTimeout(r, 100));
    await handle.cancel();
    const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
    t.true(err?.cause instanceof CancelledFailure, String(err?.cause));
  });
});

for (const [mode, title] of [
  ['finally', 'a Workflow cancellation wins over a refused session in cleanup()'],
  ['nonCancellable', 'a Workflow cancellation wins over a refused session in cleanup(), also in a non-cancellable scope'],
  ['stop', 'a cancelled scope between connect() and cleanup() wins over a refused session'],
] as const) {
  test(`Stateful MCP: ${title}`, async (t) => {
    const { createWorker, startWorkflow } = helpers(t);
    let factoryCalls = 0;
    const worker = await createWorker({
      plugins: [
        new OpenAIAgentsPlugin({
          modelProvider: new FakeModelProvider([]),
          mcpServerProviders: [
            new StatefulMCPServerProvider(
              'refusedCancel',
              () => {
                factoryCalls++;
                return memoryServer('refusedCancel', 'S');
              },
              t.context.env.nativeConnection
            ),
          ],
        }),
      ],
    });
    await worker.runUntil(async () => {
      const handle = await startWorkflow(statefulMcpCancelledAfterRefusalWorkflow, {
        args: [mode],
        workflowExecutionTimeout: '60 seconds',
      });
      // Wait until the retry of the timed-out session Activity has been refused.
      const refused = async () =>
        ((await handle.fetchHistory()).events ?? []).some(
          (e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED
        );
      for (let i = 0; i < 300 && !(await refused()); i++) await new Promise((r) => setTimeout(r, 100));
      t.true(await refused(), 'the session Activity was refused');
      if (mode === 'stop') {
        await handle.signal(resumeSignal);
        t.is(await handle.result(), 'stopped');
        return;
      }
      await handle.cancel();
      const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
      t.true(err?.cause instanceof CancelledFailure, String(err?.cause));
    });
    t.is(factoryCalls, 1);
  });
}

test('Stateful MCP: a caller still cancels a queued operation without a schedule-to-start timeout', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const calls: string[] = [];
  let release!: () => void;
  const released = new Promise<void>((resolve) => (release = resolve));
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([]),
        mcpServerProviders: [
          new StatefulMCPServerProvider(
            'sessionEnded',
            () => {
              const server = memoryServer('sessionEnded', 'S', calls);
              // The session's Worker polls only after the call is cancelled.
              server.connect = () => released;
              return server;
            },
            t.context.env.nativeConnection
          ),
        ],
      }),
    ],
  });
  await worker.runUntil(async () => {
    const handle = await startWorkflow(statefulMcpSessionEndedWorkflow, {
      args: [true],
      workflowExecutionTimeout: '30 seconds',
    });
    const callCancelled = async () =>
      ((await handle.fetchHistory()).events ?? []).some(
        (e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_CANCELED
      );
    for (let i = 0; i < 200 && !(await callCancelled()); i++) await new Promise((r) => setTimeout(r, 100));
    release();
    t.deepEqual(await handle.result(), ['cancelled', 'get=EMPTY session=S']);
  });
  t.false(calls.includes('put'));
});
