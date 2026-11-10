import type { Model, ModelProvider, ModelRequest, ModelResponse, StreamEvent } from '@openai/agents-core';
import { helpers } from '@temporalio/test-helpers';
import { OpenAIAgentsPlugin } from '..';
import { agentHooksWorkflow, customHandoffWorkflow } from './workflows/openai-agents';
import { FakeModelProvider, textResponse, toolCallResponse } from './stubs/openai-agents';
import { handoffResponse } from './stubs/openai-agents-fakes';
import { makeOpenAIAgentsTestFunction } from './helpers/openai-agents';

const test = makeOpenAIAgentsTestFunction();

/**
 * Calls fetchValue, ends after the tool result, then answers a second turn with its instructions.
 * Each answer depends only on the request, so several Workflows can share it.
 */
function hookProbeModelProvider(): ModelProvider {
  const model: Model = {
    async getResponse(request: ModelRequest): Promise<ModelResponse> {
      const input = JSON.stringify(request.input);
      if (input.includes('again')) return textResponse(`next turn saw: ${request.systemInstructions}`);
      return input.includes('function_call_result')
        ? textResponse('done')
        : toolCallResponse('fetchValue', { value: 'S-hook' });
    },
    // eslint-disable-next-line require-yield
    async *getStreamedResponse(): AsyncIterable<StreamEvent> {
      throw new Error('Streaming not supported');
    },
  };
  return { getModel: () => model };
}

test('agent.on listeners run, their context writes reach later turns, and one that throws stops the run', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);
  const plugin = new OpenAIAgentsPlugin({ modelProvider: hookProbeModelProvider() });
  const worker = await createWorker({
    plugins: [plugin],
    activities: { fetchValue: async ({ value }: { value: string }) => value },
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(agentHooksWorkflow, { workflowExecutionTimeout: '30 seconds' });
    t.deepEqual(result.events, [
      'agent_start:true',
      'agent_tool_end:fetchValue',
      'agent_end:done',
      'agent_start:true',
      'agent_end:next turn saw: hook=S-hook',
    ]);
    t.is(result.nextTurnOutput, 'next turn saw: hook=S-hook');
    // As with Node's EventEmitter, a listener that throws fails the run.
    const err = await t.throwsAsync(
      executeWorkflow(agentHooksWorkflow, { args: [true], workflowExecutionTimeout: '30 seconds' })
    );
    t.regex(String((err as any)?.cause?.message), /a guard stopped the run/);
  });
});

const deskActivities = {
  standardDesk: async () => 'standard desk',
  vipDesk: async () => 'vip desk',
};

test('A custom Handoff hands off to the agent its callback returns', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);
  const modelProvider = new FakeModelProvider([
    handoffResponse('transfer_to_desk'),
    textResponse('vip answer'),
    handoffResponse('transfer_to_desk'),
    textResponse('standard answer'),
  ]);
  const worker = await createWorker({
    plugins: [new OpenAIAgentsPlugin({ modelProvider })],
    activities: deskActivities,
  });

  const results = await worker.runUntil(async () => [
    await executeWorkflow(customHandoffWorkflow, { args: ['vip'], workflowExecutionTimeout: '30 seconds' }),
    await executeWorkflow(customHandoffWorkflow, { args: ['standard'], workflowExecutionTimeout: '30 seconds' }),
  ]);

  t.deepEqual(results, ['vip: vip answer', 'standard: standard answer']);
});
