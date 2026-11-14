import { helpers } from '@temporalio/test-helpers';
import { OpenAIAgentsPlugin } from '..';
import { copiedAgentToolApprovalWorkflow } from './workflows/openai-agents';
import { textResponse, toolCallResponse, type FakeModel } from './stubs/openai-agents';
import { makeOpenAIAgentsTestFunction, replay } from './helpers/openai-agents';

const test = makeOpenAIAgentsTestFunction();

for (const kind of ['original', 'alias', 'spread', 'namespace', 'override', 'alias-override'] as const) {
  test(`an approval inside a nested asTool() run resumes on result.state (${kind} tool)`, async (t) => {
    const { createWorker, startWorkflow } = helpers(t);
    const answered = (request: any, callId: string) =>
      Array.isArray(request.input) &&
      request.input.some((item: any) => item.type === 'function_call_result' && item.callId === callId);
    const provider = {
      getModel() {
        return {
          async getResponse(request: any) {
            if (request.systemInstructions === 'Answer as nested.') {
              return answered(request, 'call_fake_confirmAction')
                ? textResponse('nested done')
                : toolCallResponse('confirmAction', { reason: 'approved' });
            }
            if (answered(request, 'call_fake_ask_nested')) return textResponse('parent done');
            const response = toolCallResponse('ask_nested', { input: 'q' });
            if (kind === 'namespace') (response.output[0] as { namespace?: string }).namespace = 'desk';
            return response;
          },
        } as unknown as FakeModel;
      },
    };
    const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })] });

    await worker.runUntil(async () => {
      const handle = await startWorkflow(copiedAgentToolApprovalWorkflow, {
        args: [kind],
        workflowExecutionTimeout: '30 seconds',
      });
      // An Agent subclass's asTool() override runs through the deprecated alias too: the parent pauses
      // on the tool before the nested run starts.
      const overridden = kind === 'override' || kind === 'alias-override';
      t.deepEqual(await handle.result(), {
        output: 'parent done',
        executions: 1,
        resumes: overridden ? 2 : 1,
        paused: overridden ? [['ask_nested'], ['confirmAction']] : [['confirmAction']],
        pending: [],
      });
      await replay(t, handle);
    });
  });
}
