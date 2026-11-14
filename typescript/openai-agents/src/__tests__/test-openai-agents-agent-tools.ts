import { helpers } from '@temporalio/test-helpers';
import { OpenAIAgentsPlugin } from '..';
import {
  agentToolRejectedWorkflow,
  agentToolRunConfigWorkflow,
  agentToolStreamWorkflow,
  agentToolOptionsReuseWorkflow,
  type AgentToolRejection,
} from './workflows/openai-agents';
import { multiToolCallResponse, textResponse, toolCallResponse, type FakeModel } from './stubs/openai-agents';
import { streamingTextEvents } from './stubs/openai-agents-fakes';
import { makeOpenAIAgentsTestFunction, replay } from './helpers/openai-agents';

const test = makeOpenAIAgentsTestFunction();

test('agent.asTool() runs take the run model, model settings and runner defaults', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const requests: { model: string; instructions: string; modelSettings: Record<string, unknown> }[] = [];
  let leadCalls = 0;
  const provider = {
    getModel(name?: string) {
      return {
        async getResponse(request: any) {
          requests.push({
            model: name ?? '',
            instructions: request.systemInstructions,
            modelSettings: request.modelSettings,
          });
          if (request.systemInstructions === 'Answer as the expert.') return textResponse('expert answer');
          if (request.systemInstructions === 'Answer as the specialist.') return textResponse('specialist answer');
          return leadCalls++ === 0
            ? multiToolCallResponse([
                { name: 'ask_expert', args: { input: 'q1' } },
                { name: 'ask_specialist', args: { input: 'q2' } },
              ])
            : textResponse('done');
        },
      } as unknown as FakeModel;
    },
  };
  const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })] });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(agentToolRunConfigWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });
    t.is(await handle.result(), 'done');

    const first = (instructions: string) => requests.find((r) => r.instructions === instructions);
    // A nested agent with no model gets the run model; one with its own model keeps it, as natively.
    t.is(first('Answer as the expert.')?.model, 'run-model');
    t.is(first('Answer as the specialist.')?.model, 'own-model');
    for (const instructions of ['Use the tools.', 'Answer as the expert.', 'Answer as the specialist.']) {
      t.is(first(instructions)?.modelSettings.temperature, 0.33, instructions);
    }
    // toolChoice picks the outer agent's tools, so nested runs don't take it.
    t.is(first('Use the tools.')?.modelSettings.toolChoice, 'auto');
    t.is(first('Answer as the expert.')?.modelSettings.toolChoice, undefined);

    const { events } = await handle.fetchHistory();
    const modelActivities = (events ?? []).filter(
      (e) => e.activityTaskScheduledEventAttributes?.activityType?.name === 'invokeModelActivity'
    );
    t.is(modelActivities.length, 4);
    for (const e of modelActivities) {
      t.is(Number(e.activityTaskScheduledEventAttributes?.startToCloseTimeout?.seconds), 17);
      const summary = Buffer.from(e.userMetadata?.summary?.data ?? []).toString('utf-8');
      t.true(summary.includes('runner default'), summary);
    }
    await replay(t, handle);
  });
});

test('agent.asTool() setups that would run a model or sandbox in the Workflow fail before running', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const worker = await createWorker();
  const expected: Record<AgentToolRejection, [string, RegExp]> = {
    'model-object': ['AgentsWorkflowError', /Agent 'Inline' has a Model object/],
    'nested-handoff-model-object': ['AgentsWorkflowError', /Agent 'Inline' has a Model object/],
    'model-provider': ['AgentsWorkflowError', /asTool\('ask_nested'\) sets runConfig.modelProvider/],
    'raw-sandbox': ['SandboxConfigurationError', /asTool\('ask_nested'\) runConfig.sandbox.client must be created/],
    'raw-sandbox-run-options': [
      'SandboxConfigurationError',
      /asTool\('ask_nested'\) runOptions.sandbox.client must be created/,
    ],
    'undefined-tool': ['AgentsWorkflowError', /Cannot read properties of undefined/],
    'undefined-handoff': ['AgentsWorkflowError', /Cannot read properties of undefined/],
  };

  await worker.runUntil(async () => {
    for (const [kind, [type, message]] of Object.entries(expected)) {
      const handle = await startWorkflow(agentToolRejectedWorkflow, {
        args: [kind as AgentToolRejection],
        workflowExecutionTimeout: '30 seconds',
      });
      const result = await handle.result();
      t.true(result.startsWith(`${type}: `), `${kind}: ${result}`);
      t.regex(result, message, kind);
      const { events } = await handle.fetchHistory();
      t.false(
        (events ?? []).some((e) => e.activityTaskScheduledEventAttributes != null),
        `${kind}: no Activity is scheduled`
      );
    }
  });
});

for (const [title, onAfterStart] of [
  ['A nested agent.asTool() run streams to its onStream option and on() handlers', false],
  ['An on() handler registered after the run starts gets the nested agent.asTool() stream', true],
] as const) {
  test(title, async (t) => {
    const { createWorker, startWorkflow } = helpers(t);
    let leadCalls = 0;
    const provider = {
      getModel() {
        return {
          async getResponse() {
            return leadCalls++ === 0 ? toolCallResponse('ask_streamer', { input: 'q' }) : textResponse('done');
          },
          async *getStreamedResponse() {
            yield* streamingTextEvents('streamed answer');
          },
        } as unknown as FakeModel;
      },
    };
    const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })] });

    await worker.runUntil(async () => {
      const handle = await startWorkflow(agentToolStreamWorkflow, {
        args: ['Hello', onAfterStart],
        workflowExecutionTimeout: '30 seconds',
      });
      t.deepEqual(await handle.result(), { output: 'done', onStream: ['streamed answer'], on: ['streamed answer'] });
      const { events } = await handle.fetchHistory();
      const activityTypes = (events ?? [])
        .map((e) => e.activityTaskScheduledEventAttributes?.activityType?.name)
        .filter((name) => name !== undefined);
      t.deepEqual(activityTypes, ['invokeModelActivity', 'invokeModelStreamActivity', 'invokeModelActivity']);
      await replay(t, handle);
    });
  });
}

test('A nested agent.asTool() run fails to stream when model calls are local Activities', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  let streamed = false;
  const provider = {
    getModel() {
      return {
        // Answers from the request, not a call count, so a rerun local Activity gets the same answer.
        async getResponse(request: any) {
          const answered =
            Array.isArray(request.input) && request.input.some((item: any) => item.type === 'function_call_result');
          return answered ? textResponse('done') : toolCallResponse('ask_streamer', { input: 'q' });
        },
        async *getStreamedResponse() {
          streamed = true;
          yield* streamingTextEvents('streamed answer');
        },
      } as unknown as FakeModel;
    },
  };
  const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })] });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(agentToolStreamWorkflow, {
      args: ['Hello', false, true],
      workflowExecutionTimeout: '30 seconds',
    });
    t.deepEqual(await handle.result(), { output: 'done', onStream: [], on: [] });
    t.false(streamed, 'the nested model is not called');
    await replay(t, handle);
  });
});

test('agent.asTool() tools keep the options and name they were created with', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const leadTools: string[][] = [];
  const instructions: string[] = [];
  const provider = {
    getModel() {
      return {
        async getResponse(request: any) {
          instructions.push(request.systemInstructions);
          if (request.systemInstructions === 'Answer as renamed.') return textResponse('renamed answer');
          leadTools.push((request.tools ?? []).map((tool: { name: string }) => tool.name));
          return multiToolCallResponse([
            { name: 'ask_gated', args: { input: 'q1' } },
            { name: 'Original_Name', args: { input: 'q2' } },
          ]);
        },
      } as unknown as FakeModel;
    },
  };
  const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })] });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(agentToolOptionsReuseWorkflow, {
      args: ['Hello'],
      workflowExecutionTimeout: '30 seconds',
    });
    // Reusing the options object does not drop ask_gated's approval, so it pauses instead of running.
    t.deepEqual(await handle.result(), ['ask_gated']);
    // ask_early, made at module scope, is recorded, so the run isn't rejected.
    t.deepEqual(leadTools, [['ask_gated', 'ask_open', 'Original_Name', 'ask_early']]);
    t.deepEqual(instructions, ['Use the tools.', 'Answer as renamed.']);
    await replay(t, handle);
  });
});
