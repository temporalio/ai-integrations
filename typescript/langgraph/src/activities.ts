/**
 * The Worker side of the plugin: the Activities that run nodes and tasks.
 *
 * @module
 * @internal
 */

import { AsyncLocalStorageProviderSingleton } from '@langchain/core/singletons';
import { isGraphInterrupt } from '@langchain/langgraph';
import { Context } from '@temporalio/activity';
import { WorkflowStreamClient } from '@temporalio/workflow-streams/client';
import { withSources } from './activity-sources';
import { encodeValue, restoreValue } from './codec';
import type { ValueTypes } from './codec';
import type { ActivityInput, ActivityOutput, Internals, LangGraphConfigPayload, NodeSpec, TaskSpec } from './specs';

type Writer = (chunk: unknown) => void;
type Config = Record<string, unknown>;

/**
 * Rebuilds a LangGraph `RunnableConfig` from {@link LangGraphConfigPayload},
 * with a scratchpad that serves `interrupt()` the run's resume values, and
 * `signal` as the config's abort signal.
 */
export function restoreConfig(
  payload: LangGraphConfigPayload,
  writer: Writer | undefined,
  signal?: AbortSignal
): Config {
  const { scratchpad } = payload;
  const nullResume = scratchpad.nullResume ?? undefined;
  const configurable: Config = {
    ...payload.configurable,
    __pregel_scratchpad: {
      callCounter: 0,
      interruptCounter: -1,
      resume: [...scratchpad.resume],
      nullResume,
      subgraphCounter: 0,
      currentTaskInput: undefined,
      consumeNullResume: () => nullResume,
    },
    __pregel_send: () => undefined,
  };
  if (payload.hasCheckpointer) {
    // `interrupt()` only checks that the run has a checkpointer; the
    // Workflow's checkpointer saves the run's state.
    configurable.__pregel_checkpointer = true;
  }
  const config: Config = { configurable };
  if (payload.tags?.length) config.tags = payload.tags;
  if (payload.metadata && Object.keys(payload.metadata).length > 0) config.metadata = payload.metadata;
  if (payload.runName !== undefined) config.runName = payload.runName;
  if (payload.runId !== undefined) config.runId = payload.runId;
  if (payload.recursionLimit !== undefined) config.recursionLimit = payload.recursionLimit;
  if (payload.context !== undefined) config.context = payload.context;
  if (payload.executionInfo !== undefined) config.executionInfo = payload.executionInfo;
  if (writer !== undefined) config.writer = writer;
  if (signal !== undefined) config.signal = signal;
  return config;
}

function encodeOutput(output: Omit<ActivityOutput, 'value_types'>): ActivityOutput {
  const valueTypes: ValueTypes = [];
  const encoded = encodeValue(output, [], valueTypes) as Omit<ActivityOutput, 'value_types'>;
  return { ...encoded, value_types: valueTypes.length > 0 ? valueTypes : null };
}

/** The Activity that runs `spec` on the Worker. */
function activityFor(
  spec: NodeSpec | TaskSpec,
  streamingTopic: string | undefined,
  streamingBatchInterval: Internals['streamingBatchInterval']
): (input: ActivityInput) => Promise<ActivityOutput> {
  const run = async (input: ActivityInput, writer: Writer | undefined): Promise<ActivityOutput> => {
    const { args, langgraph_config } = restoreValue(
      { args: input.args, langgraph_config: input.langgraph_config },
      input.value_types
    ) as Omit<ActivityInput, 'value_types'>;
    const config = restoreConfig(langgraph_config, writer, Context.current().cancellationSignal);
    try {
      const result = await AsyncLocalStorageProviderSingleton.runWithConfig(
        config,
        () => (spec.kind === 'node' ? spec.func(args[0], config) : spec.func(...args))
      );
      return encodeOutput({ result });
    } catch (error) {
      if (isGraphInterrupt(error)) {
        return encodeOutput({ langgraph_interrupts: error.interrupts });
      }
      throw error;
    }
  };

  return async (input) => {
    if (streamingTopic === undefined) return run(input, undefined);
    const client = WorkflowStreamClient.fromWithinActivity({ batchInterval: streamingBatchInterval });
    try {
      const topic = client.topic(streamingTopic);
      return await run(input, (chunk) => topic.publish(chunk));
    } finally {
      await client[Symbol.asyncDispose]();
    }
  };
}

/**
 * The Activities for the definitions' Activity nodes and tasks, by Activity
 * name. Each is recorded with what it runs, so two Activities built from the
 * same node or task function with the same streaming options may replace each
 * other on a Worker.
 */
export function createActivities(internals: Internals): Record<string, (input: ActivityInput) => Promise<ActivityOutput>> {
  const { streamingTopic, streamingBatchInterval } = internals;
  const activities: Record<string, (input: ActivityInput) => Promise<ActivityOutput>> = {};
  for (const spec of [...internals.nodes, ...internals.tasks]) {
    if (spec.executeIn !== 'activity') continue;
    Object.assign(
      activities,
      withSources({ [spec.activityName]: activityFor(spec, streamingTopic, streamingBatchInterval) }, [
        spec.kind,
        spec.func,
        streamingTopic,
        streamingBatchInterval,
      ])
    );
  }
  return activities;
}
