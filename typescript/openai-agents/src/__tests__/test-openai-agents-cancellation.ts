import type { StreamEvent } from '@openai/agents-core';
import { WorkflowFailedError } from '@temporalio/client';
import { CancelledFailure } from '@temporalio/common';
import { temporal } from '@temporalio/proto';
import { helpers } from '@temporalio/test-helpers';
import { OpenAIAgentsPlugin } from '..';
import { activityToolCancelWorkflow, nestedToolCancelWorkflow } from './workflows/openai-agents';
import { textResponse, toolCallResponse, type FakeModel } from './stubs/openai-agents';
import { toolCallModelProvider } from './stubs/openai-agents-fakes';
import { makeOpenAIAgentsTestFunction, replay } from './helpers/openai-agents';
import EventType = temporal.api.enums.v1.EventType;

const test = makeOpenAIAgentsTestFunction();

for (const [title, stopOnFirstTool, stream, expectCancelled, maxTurns] of [
  ['Cancelling during a nested model call cancels a run that stops on the tool', true, false, true, undefined],
  ['Cancelling during a nested model call cancels a streamed run that stops on the tool', true, true, true, undefined],
  ['Cancelling during a nested model call cancels a run that calls the model again', false, false, true, undefined],
  ['Cancelling during a nested model call cancels a run that then fails', false, false, true, 1],
  ['Cancelling during a nested model call cancels a streamed run that then fails', false, true, true, 1],
  ['A run that stops on a nested tool completes when not cancelled', true, false, false, undefined],
] as const) {
  test(title, async (t) => {
    const { createWorker, startWorkflow } = helpers(t);
    let nestedStarted!: () => void;
    const nestedCalled = new Promise<void>((resolve) => (nestedStarted = resolve));
    let release!: () => void;
    const released = new Promise<void>((resolve) => (release = resolve));
    const provider = {
      getModel() {
        return {
          async getResponse(request: any) {
            if (request.systemInstructions === 'Answer as nested.') {
              nestedStarted();
              if (expectCancelled) await released;
              return textResponse('nested answer');
            }
            const toolAnswered =
              Array.isArray(request.input) && request.input.some((item: any) => item.type === 'function_call_result');
            return toolAnswered ? textResponse('lead answer') : toolCallResponse('ask_nested', { input: 'q' });
          },
          async *getStreamedResponse() {
            yield {
              type: 'response_done',
              response: {
                id: 'resp_lead',
                usage: { requests: 1, inputTokens: 10, outputTokens: 20, totalTokens: 30 },
                output: toolCallResponse('ask_nested', { input: 'q' }).output,
              },
            } as unknown as StreamEvent;
          },
        } as unknown as FakeModel;
      },
    };
    const worker = await createWorker({
      plugins: [new OpenAIAgentsPlugin({ modelProvider: provider, modelParams: { streamingTopic: 'events' } })],
    });

    await worker.runUntil(async () => {
      const handle = await startWorkflow(nestedToolCancelWorkflow, {
        args: [{ stopOnFirstTool, stream, maxTurns }],
        workflowExecutionTimeout: '30 seconds',
      });
      if (!expectCancelled) {
        t.is(await handle.result(), 'nested answer');
        await replay(t, handle);
        return;
      }
      // If the run ends before its nested model call, the assertions below report how.
      await Promise.race([nestedCalled, handle.result().catch(() => undefined)]);
      await handle.cancel();
      const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
      release();
      t.true(err?.cause instanceof CancelledFailure, String(err?.cause));
      const { events } = await handle.fetchHistory();
      t.is(events?.at(-1)?.eventType, EventType.EVENT_TYPE_WORKFLOW_EXECUTION_CANCELED);
      await replay(t, handle);
    });
  });
}

for (const stream of [false, true]) {
  test(`Cancelling during an Activity tool cancels a ${stream ? 'streamed ' : ''}run`, async (t) => {
    const { createWorker, startWorkflow } = helpers(t);
    let toolStarted!: () => void;
    const toolCalled = new Promise<void>((resolve) => (toolStarted = resolve));
    let release!: () => void;
    const released = new Promise<void>((resolve) => (release = resolve));
    const worker = await createWorker({
      plugins: [
        new OpenAIAgentsPlugin({
          modelProvider: toolCallModelProvider('holdTool', {}),
          modelParams: { streamingTopic: 'events' },
        }),
      ],
      activities: {
        holdTool: async () => {
          toolStarted();
          await released;
          return 'held';
        },
      },
    });

    await worker.runUntil(async () => {
      const handle = await startWorkflow(activityToolCancelWorkflow, {
        args: [stream],
        workflowExecutionTimeout: '30 seconds',
      });
      await Promise.race([toolCalled, handle.result().catch(() => undefined)]);
      await handle.cancel();
      const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
      release();
      t.true(err?.cause instanceof CancelledFailure, String(err?.cause));
      const { events } = await handle.fetchHistory();
      t.is(events?.at(-1)?.eventType, EventType.EVENT_TYPE_WORKFLOW_EXECUTION_CANCELED);
      await replay(t, handle);
    });
  });
}
