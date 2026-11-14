import { helpers } from '@temporalio/test-helpers';
import { OpenAIAgentsPlugin } from '..';
import { implicitModelSettingsWorkflow, type ImplicitSettingsCase } from './workflows/openai-agents';
import { textResponse, toolCallResponse, type FakeModel } from './stubs/openai-agents';
import { makeOpenAIAgentsTestFunction, replay } from './helpers/openai-agents';

const test = makeOpenAIAgentsTestFunction();

test('With no model named, the Worker adds the model settings defaults for its own default model', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const calls: { model: string; modelSettings: unknown; reasoningEffortImplicit: unknown }[] = [];
  const provider = {
    getModel(model?: string) {
      return {
        async getResponse(request: any) {
          calls.push({
            model: model ?? '',
            modelSettings: request.modelSettings,
            reasoningEffortImplicit: request._internal?.reasoningEffortImplicit,
          });
          const answered =
            Array.isArray(request.input) && request.input.some((item: any) => item.type === 'function_call_result');
          if (request.systemInstructions === 'Use the tool.') return toolCallResponse('ask_unnamed', { input: 'q' });
          if (request.systemInstructions === 'Use echo, then answer.' && !answered) return toolCallResponse('echo', {});
          return textResponse('done');
        },
      } as unknown as FakeModel;
    },
  };
  const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })] });
  const run = async (input: ImplicitSettingsCase) => {
    const handle = await startWorkflow(implicitModelSettingsWorkflow, {
      args: [input],
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(await handle.result(), 'done');
    await replay(t, handle);
  };
  const previous = process.env.OPENAI_DEFAULT_MODEL;
  try {
    await worker.runUntil(async () => {
      // The Workflow's default model is gpt-5.4-mini. A Worker whose default isn't a GPT-5 model gets
      // none of its settings, directly or in a nested run.
      process.env.OPENAI_DEFAULT_MODEL = 'gpt-4.1';
      await run({ runSettings: { temperature: 0.3 } });
      await run({ leadModel: 'gpt-5', runSettings: { temperature: 0.3 } });
      delete process.env.OPENAI_DEFAULT_MODEL;
      await run({ runSettings: { temperature: 0.3 } });
      // The agent's own settings replace the defaults, as upstream.
      await run({ runSettings: { temperature: 0.3 }, agentSettings: { temperature: 0.9 } });
      // A nested run takes the run's settings without `toolChoice` and `parallelToolCalls`.
      await run({ leadModel: 'gpt-5', runSettings: { temperature: 0.3, toolChoice: 'required', parallelToolCalls: true } });
      // Upstream clears `toolChoice` after the tool call.
      await run({ callsTool: true, runSettings: { temperature: 0.3, toolChoice: 'required' } });
      await run({ runSettings: { reasoning: { summary: 'auto' } } });
      await run({ runSettings: { reasoning: { effort: 'low' } } });
      await run({ runSettings: { temperature: 0.3 }, clearDefaults: true });
      await run({ runSettings: { temperature: 0.3 }, clearNestedDefaults: true });
      // An empty `text` keeps the default verbosity.
      await run({ runSettings: { text: {} } });
    });
  } finally {
    if (previous === undefined) delete process.env.OPENAI_DEFAULT_MODEL;
    else process.env.OPENAI_DEFAULT_MODEL = previous;
  }
  const gpt5Defaults = { reasoning: { effort: 'none' }, text: { verbosity: 'low' } };
  t.deepEqual(calls, [
    { model: '', modelSettings: { temperature: 0.3 }, reasoningEffortImplicit: false },
    // A named model keeps upstream's settings and its marker for an implicit effort.
    {
      model: 'gpt-5',
      modelSettings: { reasoning: { effort: 'low' }, text: { verbosity: 'low' }, temperature: 0.3 },
      reasoningEffortImplicit: true,
    },
    { model: '', modelSettings: { temperature: 0.3 }, reasoningEffortImplicit: false },
    { model: '', modelSettings: { ...gpt5Defaults, temperature: 0.3 }, reasoningEffortImplicit: true },
    { model: '', modelSettings: { temperature: 0.9 }, reasoningEffortImplicit: false },
    {
      model: 'gpt-5',
      modelSettings: {
        reasoning: { effort: 'low' },
        text: { verbosity: 'low' },
        temperature: 0.3,
        toolChoice: 'required',
        parallelToolCalls: true,
      },
      reasoningEffortImplicit: true,
    },
    { model: '', modelSettings: { ...gpt5Defaults, temperature: 0.3 }, reasoningEffortImplicit: true },
    { model: '', modelSettings: { ...gpt5Defaults, temperature: 0.3, toolChoice: 'required' }, reasoningEffortImplicit: true },
    { model: '', modelSettings: { ...gpt5Defaults, temperature: 0.3 }, reasoningEffortImplicit: true },
    {
      model: '',
      modelSettings: { reasoning: { effort: 'none', summary: 'auto' }, text: { verbosity: 'low' } },
      reasoningEffortImplicit: true,
    },
    { model: '', modelSettings: { reasoning: { effort: 'low' }, text: { verbosity: 'low' } }, reasoningEffortImplicit: false },
    { model: '', modelSettings: { temperature: 0.3 }, reasoningEffortImplicit: true },
    { model: '', modelSettings: { reasoning: {}, text: {}, temperature: 0.3 }, reasoningEffortImplicit: true },
    { model: '', modelSettings: gpt5Defaults, reasoningEffortImplicit: true },
  ]);
});
