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
} from '@strands-agents/sdk';
import {
  ActivityFailure,
  allHandlersFinished,
  ApplicationFailure,
  defineSignal,
  proxyActivities,
  setHandler,
  condition,
} from '@temporalio/workflow';
import { WorkflowStream } from '@temporalio/workflow-streams/workflow';
import { TemporalAgent, TemporalMCPClient, type TemporalAgentOptions, workflow as strandsWorkflow } from '../..';

export async function helloAgent(prompt: string): Promise<string> {
  const agent = new TemporalAgent({ model: 'test', printer: false });
  const result = await agent.invoke(prompt);
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
 * finished by then and `unfinished` if not. The plugin's workflow interceptor wraps handlers, which
 * makes them finish later.
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

export const approveSignal = defineSignal<[string]>('approve');

export async function interruptAgent(prompt: string): Promise<string> {
  let approval: string | null = null;
  setHandler(approveSignal, (response: string) => {
    approval = response;
  });

  const agent = new TemporalAgent({
    model: 'test',
    printer: false,
    tools: [
      strandsWorkflow.activityAsTool('deleteThing', {
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
    await condition(() => approval !== null);
    const response = approval;
    approval = null;
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
