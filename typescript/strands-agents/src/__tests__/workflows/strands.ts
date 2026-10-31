// eslint-disable-next-line import/no-unassigned-import
import '../../load-polyfills';
import { z } from 'zod';
import {
  AfterToolCallEvent,
  SlidingWindowConversationManager,
  tool,
  type BaseModelConfig,
  type InterruptResponseContent,
  type InterruptResponseContentData,
  type JSONValue,
  type Message,
} from '@strands-agents/sdk';
import {
  ActivityCancellationType,
  ActivityFailure,
  allHandlersFinished,
  ApplicationFailure,
  continueAsNew,
  defineSignal,
  defineUpdate,
  proxyActivities,
  setHandler,
  condition,
  workflowInfo,
  type ActivityOptions,
} from '@temporalio/workflow';
import { WorkflowStream } from '@temporalio/workflow-streams/workflow';
import {
  TemporalAgent,
  TemporalMCPClient,
  TemporalModel,
  type TemporalAgentOptions,
  workflow as strandsWorkflow,
} from '../..';

export async function helloAgent(prompt: string): Promise<string> {
  const agent = new TemporalAgent({ model: 'test', printer: false });
  const result = await agent.invoke(prompt);
  return result.toString();
}

/**
 * Runs one agent turn whose model activity heartbeats, with the given cancellation type, so a
 * workflow cancel during the model call reaches the model.
 */
export async function cancellableAgent(cancellationType: ActivityCancellationType): Promise<string> {
  const agent = new TemporalAgent({
    model: 'test',
    printer: false,
    activityOptions: { startToCloseTimeout: '30 seconds', heartbeatTimeout: '5 seconds', cancellationType },
  });
  const result = await agent.invoke('hello');
  return result.toString();
}

/**
 * Runs each prompt as a turn of one agent with Strands' proactive compression, then returns the
 * config the agent's model reports in workflow code.
 */
export async function compressionAgent(prompts: string[]): Promise<BaseModelConfig> {
  const agent = new TemporalAgent({
    model: 'test',
    printer: false,
    conversationManager: new SlidingWindowConversationManager({ proactiveCompression: true }),
  });
  for (const prompt of prompts) {
    await agent.invoke(prompt);
  }
  return agent.model.getConfig();
}

/**
 * Runs the prompt twice on one agent and returns each reply, or the activity error that ended the
 * turn, and the config the agent's model reports in workflow code.
 */
export async function retryConfigAgent(prompt: string): Promise<{ turns: string[]; config: BaseModelConfig }> {
  const agent = new TemporalAgent({ model: 'test', printer: false });
  const turns: string[] = [];
  for (let i = 0; i < 2; i++) {
    try {
      turns.push((await agent.invoke(prompt)).toString());
    } catch (err) {
      turns.push(`error: ${(err as ActivityFailure).cause?.message}`);
    }
  }
  return { turns, config: agent.model.getConfig() };
}

// Retry a failed model activity quickly, and not forever.
const STATEFUL_ACTIVITY_OPTIONS: ActivityOptions = {
  startToCloseTimeout: '15 seconds',
  retry: { initialInterval: '100 milliseconds', backoffCoefficient: 1, maximumAttempts: 50 },
};

/**
 * Runs two turns on one agent with the `stateful` worker model and returns the agent's message
 * count after each turn, or the error that ended the run. `models-queue` runs the model activities
 * on the `-models` task queue.
 */
export async function statefulAgent(
  mode: 'plain' | 'streaming' | 'manager' | 'models-queue'
): Promise<number[] | string> {
  void new WorkflowStream();
  const agent = new TemporalAgent({
    model: 'stateful',
    printer: false,
    activityOptions: {
      ...STATEFUL_ACTIVITY_OPTIONS,
      ...(mode === 'models-queue' ? { taskQueue: `${workflowInfo().taskQueue}-models` } : {}),
    },
    ...(mode === 'streaming' ? { streamingTopic: 'model' } : {}),
    ...(mode === 'manager' ? { conversationManager: new SlidingWindowConversationManager() } : {}),
  });
  const messagesAfterTurn: number[] = [];
  try {
    for (const prompt of ['turn one', 'turn two']) {
      await agent.invoke(prompt);
      messagesAfterTurn.push(agent.messages.length);
    }
  } catch (err) {
    return (err as Error).message;
  }
  return messagesAfterTurn;
}

export interface StatefulChatInput {
  prompts: string[];
  messages?: Message[];
  modelState?: Record<string, JSONValue>;
}

/**
 * The README's continue-as-new handoff with the `stateful` worker model: one turn per run. Returns
 * the last run's message count, or the error that ended the turn.
 */
export async function statefulChatAgent(input: StatefulChatInput): Promise<number | string> {
  const agent = new TemporalAgent({
    model: 'stateful',
    printer: false,
    activityOptions: STATEFUL_ACTIVITY_OPTIONS,
    messages: input.messages,
    modelState: input.modelState,
  });
  try {
    await agent.invoke(input.prompts[0]!);
  } catch (err) {
    return (err as Error).message;
  }
  if (input.prompts.length > 1) {
    await continueAsNew<typeof statefulChatAgent>({
      prompts: input.prompts.slice(1),
      messages: agent.messages,
      modelState: agent.modelState.getAll(),
    });
  }
  return agent.messages.length;
}

/**
 * Same as {@link helloAgent} but omits `model:` so the plugin's single-factory
 * default kicks in. Exercises the fallback that picks the only registered
 * factory's key as the implicit default.
 */
export async function defaultModelAgent(prompt: string): Promise<string> {
  const agent = new TemporalAgent({ printer: false });
  const result = await agent.invoke(prompt);
  return result.toString();
}

export async function activityToolAgent(prompt: string): Promise<string> {
  const agent = new TemporalAgent({
    model: 'test',
    printer: false,
    tools: [
      strandsWorkflow.activityAsTool('getWeather', {
        description: 'Get the weather for a city',
        inputSchema: {
          type: 'object',
          properties: {
            location: { type: 'string' },
          },
          required: ['location'],
        },
        activityOptions: { startToCloseTimeout: '10 seconds' },
      }),
    ],
  });
  const result = await agent.invoke(prompt);
  return result.toString();
}

/**
 * Runs an agent with `tools` and returns, for each tool call that failed, the text the model gets
 * and the message of the error attached to the result for hooks.
 */
async function failedToolCalls(
  prompt: string,
  tools: TemporalAgentOptions['tools']
): Promise<Array<{ text: string; error?: string }>> {
  const failed: Array<{ text: string; error?: string }> = [];
  const agent = new TemporalAgent({ model: 'test', printer: false, tools });
  agent.addHook(AfterToolCallEvent, (event) => {
    if (event.result.status === 'error') {
      const block = event.result.content[0];
      failed.push({ text: block?.type === 'textBlock' ? block.text : '', error: event.result.error?.message });
    }
  });
  await agent.invoke(prompt);
  return failed;
}

export async function failingActivityToolAgent(prompt: string): Promise<Array<{ text: string; error?: string }>> {
  return failedToolCalls(prompt, [
    strandsWorkflow.activityAsTool('getWeather', {
      description: 'Get the weather for a city',
      inputSchema: { type: 'object', properties: { location: { type: 'string' } }, required: ['location'] },
      activityOptions: { startToCloseTimeout: '10 seconds' },
    }),
  ]);
}

export async function failingMcpToolAgent(
  prompt: string,
  server: string
): Promise<Array<{ text: string; error?: string }>> {
  return failedToolCalls(prompt, [
    new TemporalMCPClient({ server, activityOptions: { startToCloseTimeout: '10 seconds' } }),
  ]);
}

export async function mcpAgent(prompt: string, server = 'testServer', cacheTools?: boolean): Promise<string> {
  const mcp = new TemporalMCPClient({
    server,
    activityOptions: { startToCloseTimeout: '10 seconds' },
    cacheTools,
  });
  const agent = new TemporalAgent({
    model: 'test',
    printer: false,
    tools: [mcp],
  });
  const result = await agent.invoke(prompt);
  return result.toString();
}

const echoTool = tool({
  name: 'echo',
  description: 'Echo back the input text',
  inputSchema: z.object({ text: z.string() }),
  // Prefix the result with a marker the stub model never emits, so a test can
  // tell the callback's output apart from the scripted tool-call input.
  callback: ({ text }) => `echoed: ${text}`,
});

export async function inWorkflowToolAgent(prompt: string): Promise<string> {
  const agent = new TemporalAgent({
    model: 'test',
    printer: false,
    tools: [echoTool],
  });
  const result = await agent.invoke(prompt);
  return result.toString();
}

export async function hooksAgent(prompt: string): Promise<string[]> {
  const firedEvents: string[] = [];
  const agent = new TemporalAgent({
    model: 'test',
    printer: false,
    tools: [echoTool],
  });
  // Sync hook — deterministic per-workflow state mutation.
  agent.addHook(AfterToolCallEvent, (event) => {
    firedEvents.push(event.toolUse.name);
  });
  // Activity-as-hook — dispatches `auditTool` activity.
  agent.addHook(
    AfterToolCallEvent,
    strandsWorkflow.activityAsHook('auditTool', {
      activityInput: (event) => event.toolUse.name,
      activityOptions: { startToCloseTimeout: '10 seconds' },
    })
  );
  await agent.invoke(prompt);
  return firedEvents;
}

const PersonInfo = z.object({
  name: z.string(),
  age: z.number(),
  occupation: z.string(),
});

export async function structuredOutputAgent(prompt: string): Promise<unknown> {
  const agent = new TemporalAgent({
    model: 'test',
    printer: false,
    structuredOutputSchema: PersonInfo,
  });
  const result = await agent.invoke(prompt);
  return result.structuredOutput;
}

/**
 * Calls an activity that fails once per message, then fails with how workflow code decoded each
 * activity failure, so a test can check how the failures are stored and decoded.
 */
export async function failActivities(messages: string[]): Promise<never> {
  const { failingActivity } = proxyActivities<{ failingActivity(message: string): Promise<void> }>({
    startToCloseTimeout: '10 seconds',
    retry: { maximumAttempts: 1 },
  });
  const decoded: unknown[] = [];
  for (const message of messages) {
    await failingActivity(message).catch((err: unknown) => {
      const cause = err instanceof ActivityFailure ? err.cause : undefined;
      decoded.push(
        cause instanceof ApplicationFailure
          ? { type: cause.type, nonRetryable: cause.nonRetryable, details: cause.details }
          : null
      );
    });
  }
  // Failing, rather than returning, also stores a workflow failure, which the bundle's failure
  // converter encodes.
  throw ApplicationFailure.create({ message: 'workflow failed', nonRetryable: true, details: decoded });
}

/** Runs each prompt as a turn of one agent, then returns the last reply and how many messages the agent kept. */
export async function turnsAgent(prompts: string[]): Promise<[string, number]> {
  const agent = new TemporalAgent({ model: 'test', printer: false });
  let reply = '';
  for (const prompt of prompts) {
    reply = (await agent.invoke(prompt)).toString();
  }
  return [reply, agent.messages.length];
}

export const finishSignal = defineSignal('finish');

/**
 * Waits for the `finish` Signal and, one microtask later, schedules `finished` if every handler has
 * finished by then and `unfinished` if not, so its history depends on when the handler finishes.
 */
export async function handlerTiming(): Promise<void> {
  const activities = proxyActivities<{ finished(): Promise<void>; unfinished(): Promise<void> }>({
    startToCloseTimeout: '10 seconds',
  });
  let signaled = false;
  setHandler(finishSignal, () => {
    signaled = true;
  });
  await condition(() => signaled);
  await Promise.resolve();
  await (allHandlersFinished() ? activities.finished() : activities.unfinished());
}

export const turnUpdate = defineUpdate<string, [string]>('turn');
export const doneSignal = defineSignal('done');

/** Runs each `turn` update as a turn of one agent, until the `done` signal. */
export async function updateTurnsAgent(): Promise<string> {
  const agent = new TemporalAgent({ model: 'test', printer: false });
  let done = false;
  setHandler(turnUpdate, async (prompt) => (await agent.invoke(prompt)).toString());
  setHandler(doneSignal, () => {
    done = true;
  });
  await condition(() => done);
  return 'finished';
}

export const approveSignal = defineSignal<[string]>('approve');

export async function interruptAgent(prompt: string, activityName = 'deleteThing'): Promise<string> {
  // One approval per pause, in the order they arrive.
  const approvals: string[] = [];
  setHandler(approveSignal, (response: string) => {
    approvals.push(response);
  });

  const agent = new TemporalAgent({
    model: 'test',
    printer: false,
    tools: [
      strandsWorkflow.activityAsTool(activityName, {
        description: 'Delete a thing',
        inputSchema: {
          type: 'object',
          properties: { name: { type: 'string' } },
          required: ['name'],
        },
        activityOptions: { startToCloseTimeout: '10 seconds' },
      }),
    ],
  });

  let result = await agent.invoke(prompt);
  while (result.stopReason === 'interrupt') {
    await condition(() => approvals.length > 0);
    const response = approvals.shift() as string;
    const responses: InterruptResponseContentData[] = (result.interrupts ?? []).map((i) => ({
      type: 'interruptResponse',
      interruptResponse: { interruptId: i.id, response },
    }));
    result = await agent.invoke(responses as InterruptResponseContent[]);
  }
  return result.toString();
}

export async function streamingAgent(prompt: string): Promise<string> {
  // Constructing the stream installs the publish/poll handlers the client uses
  // to drain the `model` topic; nothing in this function reads from it.
  void new WorkflowStream();
  const agent = new TemporalAgent({
    model: 'test',
    printer: false,
    streamingTopic: 'model',
  });
  const result = await agent.invoke(prompt);
  return result.toString();
}

/**
 * Streams one model call whose activity heartbeats and waits for its cancellation to complete, so
 * a workflow cancel reaches the model call and the workflow ends with the activity's result.
 */
export async function cancellableModelCall(streaming: boolean): Promise<string[]> {
  if (streaming) void new WorkflowStream();
  const model = new TemporalModel({
    modelName: 'test',
    streamingTopic: streaming ? 'model' : undefined,
    activityOptions: {
      startToCloseTimeout: '30 seconds',
      heartbeatTimeout: '5 seconds',
      cancellationType: ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
    },
  });
  const types: string[] = [];
  for await (const event of model.stream([])) {
    types.push(event.type);
  }
  return types;
}
