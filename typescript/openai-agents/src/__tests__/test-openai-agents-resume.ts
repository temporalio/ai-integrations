import { helpers } from '@temporalio/test-helpers';
import { OpenAIAgentsPlugin } from '..';
import { sharedAgentApprovalWorkflow } from './workflows/openai-agents';
import { textResponse, toolCallResponse, type FakeModel } from './stubs/openai-agents';
import { handoffResponse } from './stubs/openai-agents-fakes';
import { makeOpenAIAgentsTestFunction, replay } from './helpers/openai-agents';

const test = makeOpenAIAgentsTestFunction();

for (const [title, legacyFirstRun, nestedCall] of [
  ['a handoff to an agent that is also an asTool() source resumes after continueAsNew', false, 'handoff'],
  ['an approval inside a nested asTool() run resumes after continueAsNew', false, 'tool'],
  ['state saved before agent.asTool() replaced a function tool resumes', true, 'handoff'],
] as const) {
  test(title, async (t) => {
    const { createWorker, startWorkflow } = helpers(t);
    const calls: { instructions: string; model: string }[] = [];
    const answered = (request: any, callId: string) =>
      Array.isArray(request.input) &&
      request.input.some((item: any) => item.type === 'function_call_result' && item.callId === callId);
    const provider = {
      getModel(name?: string) {
        return {
          async getResponse(request: any) {
            calls.push({ instructions: request.systemInstructions, model: name ?? '' });
            if (request.systemInstructions === 'Answer as nested.') {
              return answered(request, 'call_fake_confirmAction')
                ? textResponse('nested done')
                : toolCallResponse('confirmAction', { reason: 'approved' });
            }
            if (answered(request, 'call_fake_ask_nested')) return textResponse('parent done');
            return nestedCall === 'handoff'
              ? handoffResponse('transfer_to_Nested')
              : toolCallResponse('ask_nested', { input: 'q' });
          },
        } as unknown as FakeModel;
      },
    };
    const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })] });

    await worker.runUntil(async () => {
      const handle = await startWorkflow(sharedAgentApprovalWorkflow, {
        args: [{ legacyFirstRun }],
        workflowExecutionTimeout: '30 seconds',
      });
      const result = await handle.result();
      t.is(result.savedIdentity, nestedCall === 'handoff' ? 'Nested' : 'Parent');
      t.is(result.output, nestedCall === 'handoff' ? 'nested done' : 'parent done');
      t.is(result.executions, 1, 'the approved action runs once');
      // Nested sets its own model, which wins over runConfig.model wherever it runs, as upstream.
      const nestedModels = calls.filter((c) => c.instructions === 'Answer as nested.').map((c) => c.model);
      t.deepEqual(nestedModels, ['own-model', 'own-model']);
      await replay(t, handle);
    });
  });
}
