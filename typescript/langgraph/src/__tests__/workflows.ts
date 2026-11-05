/**
 * The test Workflows. Each takes the key of its definitions in `registry`,
 * so one bundle serves every test.
 */

import * as uuid from '@langchain/core/utils/uuid';
import { Command, MemorySaver } from '@langchain/langgraph';
import { condition, continueAsNew, defineSignal, setHandler, workflowInfo } from '@temporalio/workflow';
import { WorkflowStream } from '@temporalio/workflow-streams/workflow';
import { cache } from '../workflow';
import type { TaskCache } from '../workflow';
import { registry } from './graphs';

 

export const ackDone = defineSignal('ack_done');

function waitForAck(): Promise<void> {
  let acked = false;
  setHandler(ackDone, () => {
    acked = true;
  });
  return condition(() => acked);
}

export interface RunOptions {
  context?: unknown;
}

/** Compiles and invokes a registered graph. */
export async function runGraph(key: string, graphName: string, input: unknown, options: RunOptions = {}): Promise<any> {
  const app = (registry[key]!.graph(graphName) as any).compile();
  return app.invoke(input, options.context === undefined ? undefined : { context: options.context });
}

/** Invokes a registered entrypoint. */
export async function runEntrypoint(key: string, name: string, input: unknown, options: RunOptions = {}): Promise<any> {
  const app = registry[key]!.entrypoint(name) as any;
  return app.invoke(input, options.context === undefined ? undefined : { context: options.context });
}

/** Interrupts, checks the interrupt, then resumes. */
export async function interruptWorkflow(input: string, inspect: 'result' | 'state'): Promise<any> {
  const app = (registry.interruptGraph!.graph('my-graph') as any).compile({ checkpointer: new MemorySaver() });
  const config = { configurable: { thread_id: '1' } };
  const result = await app.invoke({ value: input }, config);
  if (inspect === 'result') {
    if (result.__interrupt__?.[0]?.value !== 'Continue?') {
      throw new Error(`Unexpected interrupt result: ${JSON.stringify(result)}`);
    }
  } else {
    const state = await app.getState(config);
    const interrupts = state.tasks.flatMap((task: any) => task.interrupts);
    if (JSON.stringify(state.values) !== JSON.stringify({ value: '' }) || interrupts.length !== 1) {
      throw new Error(`Unexpected state: ${JSON.stringify(state)}`);
    }
    if (interrupts[0].value !== 'Continue?') throw new Error(`Unexpected interrupt: ${JSON.stringify(interrupts)}`);
  }
  return app.invoke(new Command({ resume: 'yes' }), config);
}

/** Streams a graph's custom writes to the `tokens` topic. */
export async function streamingWorkflow(key: string, input: string): Promise<string> {
  const stream = new WorkflowStream();
  const acked = waitForAck();
  const app = (registry[key]!.graph('streaming-ws', { stream }) as any).compile();
  const result = await app.invoke({ value: input });
  await acked;
  return result.value;
}

/** Publishes each `stream()` chunk from the Workflow. */
export async function astreamPublishWorkflow(input: string): Promise<string> {
  const stream = new WorkflowStream();
  const acked = waitForAck();
  const topic = stream.topic('astream');
  const app = (registry.astreamPublish!.graph('astream-publish') as any).compile();
  for await (const chunk of await app.stream({ value: input }, { streamMode: 'updates' })) {
    topic.publish(chunk);
  }
  topic.publish({ done: true });
  await acked;
  return 'done';
}

/** Runs the graph with a checkpointer, then continues as new with its state, until the value has 3 characters. */
export async function continueAsNewWorkflow(values: { value: string }): Promise<{ value: string }> {
  const app = (registry.continueAsNew!.graph('my-graph') as any).compile({ checkpointer: new MemorySaver() });
  const config = { configurable: { thread_id: '1' } };
  await app.updateState(config, values);
  await app.invoke(values, config);
  if (values.value.length < 3) {
    const state = await app.getState(config);
    await continueAsNew<typeof continueAsNewWorkflow>(state.values);
  }
  return values;
}

export interface CachedInput {
  key: string;
  graphName: string;
  input: unknown;
  cache?: TaskCache | null;
  phase?: number;
  phases: number;
}

/** Runs a graph, then continues as new with the task cache until `phases`. */
export async function cachedGraphWorkflow(input: CachedInput): Promise<{ state: any; cache: TaskCache | null }> {
  const phase = input.phase ?? 1;
  const app = (registry[input.key]!.graph(input.graphName, { cache: input.cache }) as any).compile();
  const state = await app.invoke(input.input);
  if (phase < input.phases) {
    await continueAsNew<typeof cachedGraphWorkflow>({ ...input, cache: cache(), phase: phase + 1 });
  }
  return { state, cache: cache() };
}

export interface FunctionalCachedInput {
  value: number;
  cache?: TaskCache | null;
  phase?: number;
}

/** Runs the entrypoint three times, continuing as new with the task cache in between. */
export async function continueAsNewFunctionalWorkflow(input: FunctionalCachedInput): Promise<any> {
  const phase = input.phase ?? 1;
  const app = registry.continueAsNewFunctional!.entrypoint('e2e_continue_as_new_functional', {
    cache: input.cache,
  }) as any;
  const result = await app.invoke(input.value);
  if (phase < 3) {
    await continueAsNew<typeof continueAsNewFunctionalWorkflow>({ value: input.value, cache: cache(), phase: phase + 1 });
  }
  return result;
}

/** Phase 1 runs three tasks, phase 2 all five, the first three from the cache. */
export async function partialExecutionWorkflow(input: FunctionalCachedInput): Promise<any> {
  const phase = input.phase ?? 1;
  const app = registry.partialExecution!.entrypoint('e2e_partial_execution', { cache: input.cache }) as any;
  if (phase === 1) {
    await app.invoke({ value: input.value, stopAfter: 3 });
    await continueAsNew<typeof partialExecutionWorkflow>({ value: input.value, cache: cache(), phase: 2 });
  }
  return app.invoke({ value: input.value, stopAfter: 5 });
}

export interface CryptoProbe {
  v4: string;
  v7: string;
  direct: string;
  bytes: string;
  echoed: string;
}

/**
 * Draws IDs through LangChain's `uuid` and the `crypto` global, then sends them
 * through an Activity graph, whose result comes from history on replay.
 */
export async function cryptoShimProbe(): Promise<CryptoProbe> {
  const bytes = crypto.getRandomValues(new Uint8Array(8));
  const ids = {
    v4: uuid.v4(),
    v7: uuid.v7(),
    direct: crypto.randomUUID(),
    bytes: [...bytes].map((byte) => byte.toString(16).padStart(2, '0')).join(''),
  };
  const app = (registry['twoNodes.activity']!.graph('my-graph') as any).compile();
  const result = await app.invoke({ value: JSON.stringify(ids) });
  return { ...ids, echoed: result.value };
}

/** A Functional API task interrupts; the Workflow checks the interrupt and resumes it. */
export async function interruptFunctionalWorkflow(input: string): Promise<any> {
  const app = registry.interruptFunctional!.entrypoint('v2_interrupt') as any;
  app.checkpointer = new MemorySaver();
  const config = { configurable: { thread_id: workflowInfo().workflowId } };
  await app.invoke(input, config);
  const state = await app.getState(config);
  const interrupts = state.tasks.flatMap((task: any) => task.interrupts);
  if (interrupts.length !== 1 || interrupts[0].value !== 'Do you approve?') {
    throw new Error(`Unexpected interrupts: ${JSON.stringify(interrupts)}`);
  }
  return app.invoke(new Command({ resume: 'approved' }), config);
}
