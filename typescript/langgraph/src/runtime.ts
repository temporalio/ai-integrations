/**
 * The Workflow side of the plugin: patches this Workflow's copies of the
 * registered graphs and entrypoints so that nodes and tasks run as Activities
 * or inline in the Workflow.
 *
 * This module is part of the Workflow-sandbox import graph: it must not import
 * any Worker-only module.
 *
 * @module
 * @internal
 */

import { GraphInterrupt, getConfig } from '@langchain/langgraph';
import { AsyncLocalStorageProviderSingleton } from '@langchain/core/singletons';
import { log, proxyActivities, setCurrentDetails } from '@temporalio/workflow';
import type { ActivityOptions } from '@temporalio/workflow';
import { encodeValue, restoreValue } from './codec';
import type { ValuePath, ValueTypes } from './codec';
import { runState } from './run-state';
import { cacheKey } from './task-cache';
import { splitMetadata } from './options';
import type {
  ActivityInput,
  ActivityOutput,
  Internals,
  LangGraphConfigPayload,
  NodeSpec,
  RunnableCallableLike,
  TaskFunc,
  TaskSpec,
} from './specs';

type Config = Record<string, any>;
type CallFn = (func: TaskFunc, name: string, args: unknown[], options: unknown) => unknown;

/**
 * `configurable` keys copied to the Activity: plain values LangGraph reads
 * there. The same keys as the Python plugin's `strip_runnable_config`, plus
 * `__pregel_previous`, where LangGraph.js keeps what Python carries as
 * `Runtime.previous`.
 */
const CONFIGURABLE_KEYS = [
  'checkpoint_ns',
  'checkpoint_id',
  'checkpoint_map',
  'thread_id',
  '__pregel_task_id',
  '__pregel_resuming',
  '__pregel_durability',
  '__pregel_previous',
] as const;

const CALL_KEY = '__pregel_call';
const DISPATCHING = Symbol.for('temporalio.langgraph.dispatching');

/** Wraps a run's task scheduler so the plugin's tasks run as Activities or inline. */
type DispatchingCall = CallFn & { [DISPATCHING]?: true };

/** Logged once per run, as the Python plugin does. */
const STORE_WARNING =
  'LangGraph Store passed via compile({ store }) / entrypoint({ store }) is not accessible inside activity nodes ' +
  "and tasks: the Store object isn't serializable across the Activity boundary, and Activities may run on a " +
  'different Worker than the Workflow. Use a backend-backed store (Postgres/Redis) configured on each Worker if ' +
  'you need shared memory, or use Workflow state for per-run memory.';

/** The serializable subset of a LangGraph `RunnableConfig`. */
export function serializeConfig(config: Config | undefined): LangGraphConfigPayload {
  const orig = config ?? {};
  const configurable: Config = orig.configurable ?? {};
  const scratchpad: Config | undefined = configurable.__pregel_scratchpad;
  const subset: Record<string, unknown> = {};
  for (const key of CONFIGURABLE_KEYS) {
    if (configurable[key] !== undefined) subset[key] = configurable[key];
  }
  const payload: LangGraphConfigPayload = {
    tags: [...(orig.tags ?? [])],
    metadata: { ...(orig.metadata ?? {}) },
    configurable: subset,
    scratchpad: { resume: [...(scratchpad?.resume ?? [])], nullResume: scratchpad?.nullResume ?? null },
    hasCheckpointer: Boolean(configurable.__pregel_checkpointer),
  };
  if (orig.runName !== undefined) payload.runName = orig.runName;
  if (orig.runId !== undefined) payload.runId = orig.runId;
  if (orig.recursionLimit !== undefined) payload.recursionLimit = orig.recursionLimit;
  if (orig.context !== undefined) payload.context = orig.context;
  if (orig.executionInfo !== undefined) payload.executionInfo = orig.executionInfo;
  return payload;
}

function warnStore(config: Config | undefined): void {
  if (config?.store == null) return;
  const state = runState();
  if (state.warnedStore) return;
  state.warnedStore = true;
  log.warn(STORE_WARNING);
}

function lookup(key: string): { found: boolean; value?: unknown } {
  const { cache } = runState();
  if (!(key in cache)) return { found: false };
  return { found: true, value: restoreValue(cache[key], cache[`${key}:types`] as ValueTypes | undefined) };
}

function store(key: string, value: unknown, valueTypes: ValueTypes | null): void {
  const { cache } = runState();
  cache[key] = value;
  if (valueTypes) cache[`${key}:types`] = valueTypes;
}

/** Runs a node or task as an Activity, or returns its cached result. */
async function executeActivity(spec: NodeSpec | TaskSpec, args: unknown[], config: Config | undefined): Promise<unknown> {
  warnStore(config);
  const langgraphConfig = serializeConfig(config);
  const key = cacheKey(spec.activityName, args, langgraphConfig.context);
  const cached = lookup(key);
  if (cached.found) return cached.value;

  const valueTypes: ValueTypes = [];
  const encoded = encodeValue({ args, langgraph_config: langgraphConfig }, [], valueTypes) as Omit<
    ActivityInput,
    'value_types'
  >;
  const input: ActivityInput = { ...encoded, value_types: valueTypes.length > 0 ? valueTypes : null };
  const options: ActivityOptions = { ...spec.activityOptions };
  const summary = spec.summaryFn?.(args);
  if (summary) options.summary = summary;
  const activities = proxyActivities<Record<string, (input: ActivityInput) => Promise<ActivityOutput>>>(options);
  const output = await activities[spec.activityName]!(input);
  const restored = restoreValue(output, output.value_types) as ActivityOutput;
  if (restored.langgraph_interrupts != null) {
    throw new GraphInterrupt(restored.langgraph_interrupts as never);
  }
  const resultTypes = (output.value_types ?? [])
    .filter(([path]) => path[0] === 'result')
    .map(([path, typeId]): [ValuePath, string] => [path.slice(1), typeId]);
  store(key, output.result ?? null, resultTypes.length > 0 ? resultTypes : null);
  return restored.result;
}

/** A scheduler that runs the plugin's tasks as Activities or inline, then defers to `original`. */
function dispatchingCall(internals: Internals, original: CallFn): DispatchingCall {
  const byFunc = new Map<TaskFunc, TaskSpec>(internals.tasks.map((spec) => [spec.func, spec]));
  const call: DispatchingCall = (func, name, args, options) => {
    const spec = byFunc.get(func);
    if (spec === undefined) return original(func, name, args, options);
    const replacement: TaskFunc =
      spec.executeIn === 'activity'
        ? (...taskArgs) => executeActivity(spec, taskArgs, getConfig() as Config | undefined)
        : (...taskArgs) => runInline(internals, spec, taskArgs, getConfig() as Config | undefined, () => spec.func(...taskArgs));
    return original(replacement, name, args, options);
  };
  call[DISPATCHING] = true;
  return call;
}

/** `config` with the plugin's task scheduler and, for a Workflow node or task, its stream writer. */
export function patchConfig(
  internals: Internals,
  config: Config | undefined,
  spec: NodeSpec | TaskSpec | undefined
): Config {
  const configurable: Config = { ...(config?.configurable ?? {}) };
  const call = configurable[CALL_KEY] as DispatchingCall | undefined;
  if (typeof call === 'function') {
    if (!call[DISPATCHING]) configurable[CALL_KEY] = dispatchingCall(internals, call);
  } else if (internals.tasks.length > 0) {
    // Without the scheduler the tasks would run inline in the Workflow, doing their I/O there.
    throw new Error(
      `LangGraph didn't provide its task scheduler ('configurable.${CALL_KEY}'), so LangGraphPlugin can't run ` +
        'tasks as Activities. This version of @langchain/langgraph is not supported by @temporalio/langgraph.'
    );
  }
  const patched: Config = { ...config, configurable };
  const topic = internals.streamingTopic;
  if (spec !== undefined && topic !== undefined) {
    const stream = runState().stream;
    if (stream !== undefined) {
      const handle = stream.topic(topic);
      patched.writer = (chunk: unknown) => handle.publish(chunk);
    }
  }
  return patched;
}

/** Runs `run` in the Workflow under the patched config. */
function runInline(
  internals: Internals,
  spec: NodeSpec | TaskSpec | undefined,
  args: unknown[],
  config: Config | undefined,
  run: (config: Config) => unknown
): Promise<unknown> {
  if (spec?.summaryFn !== undefined) {
    // Always written, so this node never shows a stale summary from an earlier one.
    setCurrentDetails(spec.summaryFn(args) || '');
  }
  const patched = patchConfig(internals, config, spec);
  return Promise.resolve(AsyncLocalStorageProviderSingleton.runWithConfig(patched, () => run(patched)));
}

/** Patches this Workflow's copies of the registered graphs and entrypoints, once. */
export function install(internals: Internals): void {
  if (internals.installed) return;
  internals.installed = true;

  for (const spec of internals.nodes) {
    const node = internals.graphs[spec.graphName]!.nodes[spec.nodeName]!;
    const { rest } = splitMetadata(node.metadata);
    // Pregel merges node metadata into the node's `config.metadata`, so the
    // plugin's options must not stay there.
    node.metadata = rest;
    const runnable = node.runnable as RunnableCallableLike;
    runnable.func =
      spec.executeIn === 'activity'
        ? (state, config) => executeActivity(spec, [state], config)
        : (state, config) => runInline(internals, spec, [state], config, (patched) => spec.func(state, patched));
  }

  for (const [name, program] of Object.entries(internals.entrypoints)) {
    const bound = program.nodes[name]!.bound as RunnableCallableLike;
    const func = bound.func;
    bound.func = (input, config) => runInline(internals, undefined, [input], config, (patched) => func(input, patched));
  }
}
