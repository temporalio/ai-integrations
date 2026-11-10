/**
 * `defineLangGraph()`: the graphs, entrypoints and tasks a Worker runs, shared
 * by the Worker (which registers their Activities) and the Workflow sandbox
 * (which runs the graphs).
 *
 * This module is part of the Workflow-sandbox import graph: it must not import
 * any Worker-only module.
 *
 * @module
 */

import { AsyncLocalStorageProviderSingleton } from '@langchain/core/singletons';
import { inWorkflowContext } from '@temporalio/workflow';
import type { Duration } from '@temporalio/common';
import type { WorkflowStream } from '@temporalio/workflow-streams/workflow';
import { mergeOptions, resolveOptions, splitMetadata, validateDefaults } from './options';
import type { LangGraphActivityOptions, NodeOptions } from './options';
import { runState } from './run-state';
import type { TaskCache } from './task-cache';
import { install } from './runtime';
import type {
  Internals,
  NodeSpec,
  PregelLike,
  RunnableCallableLike,
  StateGraphLike,
  TaskFunc,
  TaskSpec,
} from './specs';

/** A Functional API task, as `task()` returns it. */
export type LangGraphTask = (...args: any[]) => unknown;

/** Options for {@link defineLangGraph}. */
export interface LangGraphDefinitionOptions<
  Graphs extends Record<string, object> = Record<string, object>,
  Entrypoints extends Record<string, object> = Record<string, object>,
> {
  /**
   * Graph API `StateGraph`s, keyed by name. Each node's `metadata` must set
   * `executeIn` (`'activity'` or `'workflow'`) and may set any
   * {@link LangGraphActivityOptions} field; the plugin removes those keys
   * before LangGraph exposes the metadata to the node.
   */
  graphs?: Graphs;
  /** Functional API `entrypoint()` programs, keyed by name. */
  entrypoints?: Entrypoints;
  /** Functional API tasks the plugin runs as Activities or inline in the Workflow. */
  tasks?: LangGraphTask[];
  /**
   * Options for each of {@link tasks}, keyed by task name. Each entry must set
   * `executeIn`. The Functional API has no per-task metadata, so options go here.
   */
  activityOptions?: Record<string, NodeOptions>;
  /**
   * Options for every node and task, overridden per node (`metadata`) or task
   * ({@link activityOptions}). `executeIn` can't be set here.
   */
  defaultActivityOptions?: LangGraphActivityOptions;
  /**
   * When set, LangGraph's custom stream writer (`config.writer`, or
   * `getWriter()`) in a node or task publishes to this topic of the Workflow's
   * `WorkflowStream`. Workflows pass their stream to `graph()` or `entrypoint()`.
   *
   * @experimental Streaming support is experimental and may change without notice.
   */
  streamingTopic?: string;
  /**
   * How often an Activity's stream client flushes buffered publishes in one
   * signal. Default 100ms. No effect on Workflow nodes and tasks, which append
   * to the stream directly.
   *
   * @experimental Streaming support is experimental and may change without notice.
   */
  streamingBatchInterval?: Duration;
}

/** Options for {@link LangGraphDefinitions.graph} and {@link LangGraphDefinitions.entrypoint}. */
export interface RunOptions {
  /**
   * Task results from a previous run's {@link cache}, so that nodes and tasks
   * completed before continue-as-new are not executed again.
   */
  cache?: TaskCache | null;
  /** The Workflow's stream. Required when the definitions set `streamingTopic`. */
  stream?: WorkflowStream;
}

const internalsByDefinitions = new WeakMap<object, Internals>();

/** The definitions' registered nodes, tasks and options. */
export function internalsOf(definitions: LangGraphDefinitions): Internals {
  const internals = internalsByDefinitions.get(definitions);
  if (internals === undefined) {
    throw new TypeError('Expected the object returned by defineLangGraph().');
  }
  return internals;
}

/** The graphs, entrypoints and tasks returned by {@link defineLangGraph}. */
export class LangGraphDefinitions<
  Graphs extends Record<string, object> = Record<string, object>,
  Entrypoints extends Record<string, object> = Record<string, object>,
> {
  /** @internal */
  constructor(internals: Internals) {
    internalsByDefinitions.set(this, internals);
  }

  /**
   * A registered `StateGraph`, from a Workflow. Compile it as usual; its nodes
   * run as configured.
   */
  graph<Name extends keyof Graphs & string>(name: Name, options?: RunOptions): Graphs[Name] {
    const internals = this.prepare('graph', options);
    if (!(name in internals.graphs)) {
      throw new RangeError(`Graph '${name}' not found. Available graphs: ${JSON.stringify(Object.keys(internals.graphs))}`);
    }
    return internals.graphs[name] as unknown as Graphs[Name];
  }

  /** A registered `entrypoint()` program, from a Workflow. */
  entrypoint<Name extends keyof Entrypoints & string>(name: Name, options?: RunOptions): Entrypoints[Name] {
    const internals = this.prepare('entrypoint', options);
    if (!(name in internals.entrypoints)) {
      throw new RangeError(
        `Entrypoint '${name}' not found. Available entrypoints: ${JSON.stringify(Object.keys(internals.entrypoints))}`
      );
    }
    return internals.entrypoints[name] as unknown as Entrypoints[Name];
  }

  private prepare(method: string, options: RunOptions | undefined): Internals {
    if (!inWorkflowContext() || !runState().pluginActive) {
      throw new Error(`${method}() must be called from inside a Workflow running under LangGraphPlugin.`);
    }
    const internals = internalsOf(this);
    const state = runState();
    if (options?.cache) Object.assign(state.cache, options.cache);
    if (options?.stream) state.stream = options.stream;
    if (internals.streamingTopic !== undefined && state.stream === undefined) {
      throw new Error(
        `LangGraphPlugin was configured with streamingTopic '${internals.streamingTopic}', but this Workflow ` +
          `didn't pass its WorkflowStream. Construct a WorkflowStream at the start of the Workflow and pass it ` +
          `as ${method}(name, { stream }).`
      );
    }
    install(internals);
    return internals;
  }
}

/**
 * The task results of this run, for continue-as-new: pass them to the next
 * run's `graph(name, { cache })` or `entrypoint(name, { cache })`. `null` when
 * no results are cached.
 */
export function cache(): TaskCache | null {
  const { cache } = runState();
  return Object.keys(cache).length > 0 ? { ...cache } : null;
}

interface CapturedTask {
  func: TaskFunc;
  name: string;
  retry: unknown;
}

/** Reads a task's function, name and retry policy without running it. */
function introspectTask(task: LangGraphTask): CapturedTask {
  if (typeof task !== 'function') {
    throw new TypeError('Each of tasks must be a function returned by task().');
  }
  let captured: CapturedTask | undefined;
  const capture = (func: TaskFunc, name: string, _args: unknown[], options: { retry?: unknown } | undefined) => {
    captured = { func, name, retry: options?.retry };
  };
  const result = AsyncLocalStorageProviderSingleton.runWithConfig(
    { configurable: { __pregel_call: capture } },
    () => {
      try {
        return task();
      } catch {
        return undefined;
      }
    }
  );
  void Promise.resolve(result).catch(() => undefined);
  if (captured === undefined) {
    throw new TypeError(
      'Each of tasks must be a function returned by task() from @langchain/langgraph, called with ' +
        '@langchain/langgraph loaded (it initializes the AsyncLocalStorage LangGraph needs).'
    );
  }
  return captured;
}

function isRunnableCallable(runnable: unknown): runnable is RunnableCallableLike {
  const candidate = runnable as Partial<RunnableCallableLike> | null;
  return (
    typeof candidate === 'object' &&
    candidate !== null &&
    typeof candidate.func === 'function' &&
    Array.isArray(candidate.lc_namespace) &&
    candidate.lc_namespace[0] === 'langgraph'
  );
}

function nodeSpecs(
  graphs: Record<string, StateGraphLike>,
  defaults: LangGraphActivityOptions | undefined
): NodeSpec[] {
  const specs: NodeSpec[] = [];
  for (const [graphName, graph] of Object.entries(graphs)) {
    if (graph?._nodeDefaults?.retryPolicy !== undefined) {
      throw new TypeError(
        `Graph ${graphName} has a LangGraph default retryPolicy set. Use Temporal activity options instead, ` +
          `e.g. pass retry: {...} via defaultActivityOptions or in a node's metadata.`
      );
    }
    for (const [nodeName, node] of Object.entries(graph.nodes)) {
      const label = `Node ${graphName}.${nodeName}`;
      if (node.isErrorHandler) {
        throw new TypeError(
          `${label} is a LangGraph errorHandler, which LangGraphPlugin doesn't support. Handle errors in the node, ` +
            'or use Temporal retry options.'
        );
      }
      if (node.retryPolicy !== undefined) {
        throw new TypeError(
          `${label} has a LangGraph retryPolicy set. Use Temporal activity options instead, e.g. pass ` +
            `retry: {...} via defaultActivityOptions or in the node's metadata.`
        );
      }
      if (!isRunnableCallable(node.runnable)) {
        throw new TypeError(`${label} must be a function (a RunnableCallable), not another Runnable.`);
      }
      const { options } = splitMetadata(node.metadata);
      if (!('executeIn' in options)) {
        throw new TypeError(
          `${label} is missing required 'executeIn' in metadata. Set it to 'activity' or 'workflow'.`
        );
      }
      const activityName = `${graphName}.${nodeName}`;
      specs.push({
        kind: 'node',
        graphName,
        nodeName,
        activityName,
        func: node.runnable.func,
        ...resolveOptions(activityName, mergeOptions(defaults, options)),
      });
    }
  }
  return specs;
}

function taskSpecs(
  tasks: LangGraphTask[],
  activityOptions: Record<string, NodeOptions> | undefined,
  defaults: LangGraphActivityOptions | undefined
): TaskSpec[] {
  const specs: TaskSpec[] = [];
  const names = new Set<string>();
  for (const task of tasks) {
    const { func, name, retry } = introspectTask(task);
    if (names.has(name)) throw new TypeError(`Task name '${name}' is registered more than once.`);
    names.add(name);
    if (retry !== undefined) {
      throw new TypeError(
        `Task ${name} has a LangGraph retry policy set. Use Temporal activity options instead, e.g. pass ` +
          `retry: {...} via defaultActivityOptions or activityOptions['${name}'].`
      );
    }
    const options = (activityOptions?.[name] ?? {}) as unknown as Record<string, unknown>;
    if (!('executeIn' in options)) {
      throw new TypeError(
        `Task ${name} is missing required 'executeIn' in activityOptions['${name}']. Set it to 'activity' or ` +
          `'workflow'.`
      );
    }
    specs.push({ kind: 'task', name, activityName: name, func, ...resolveOptions(name, mergeOptions(defaults, options)) });
  }
  return specs;
}

function checkEntrypoints(entrypoints: Record<string, PregelLike>): void {
  for (const [name, program] of Object.entries(entrypoints)) {
    const nodes = Object.values(program?.nodes ?? {});
    if (nodes.length !== 1 || !isRunnableCallable(nodes[0]!.bound)) {
      throw new TypeError(`Entrypoint ${name} must be a program returned by entrypoint().`);
    }
    if (!(name in program.nodes)) {
      throw new TypeError(
        `Entrypoint ${name} must be registered under its own name, '${Object.keys(program.nodes)[0]}'.`
      );
    }
  }
}

/**
 * Defines the LangGraph graphs, entrypoints and tasks a Worker runs.
 *
 * Call it in a module both the Worker and the Workflows import: pass the
 * result to `LangGraphPlugin`, and use its `graph()` and `entrypoint()` in
 * Workflows.
 *
 * @throws TypeError if a node or task is missing `executeIn`, sets an invalid
 * one, sets a LangGraph retry policy, or is not a plain function.
 *
 * @experimental The LangGraph plugin is an experimental feature; APIs may change without notice.
 */
export function defineLangGraph<
  Graphs extends Record<string, object> = Record<string, never>,
  Entrypoints extends Record<string, object> = Record<string, never>,
>(options: LangGraphDefinitionOptions<Graphs, Entrypoints> = {}): LangGraphDefinitions<Graphs, Entrypoints> {
  validateDefaults(options.defaultActivityOptions);
  const graphs = (options.graphs ?? {}) as unknown as Record<string, StateGraphLike>;
  const entrypoints = (options.entrypoints ?? {}) as unknown as Record<string, PregelLike>;
  checkEntrypoints(entrypoints);
  const internals: Internals = {
    graphs,
    entrypoints,
    nodes: nodeSpecs(graphs, options.defaultActivityOptions),
    tasks: taskSpecs(options.tasks ?? [], options.activityOptions, options.defaultActivityOptions),
    streamingTopic: options.streamingTopic,
    streamingBatchInterval: options.streamingBatchInterval ?? '100ms',
    installed: false,
  };
  const names = new Set<string>();
  for (const spec of [...internals.nodes, ...internals.tasks]) {
    if (names.has(spec.activityName)) {
      throw new TypeError(`Activity name '${spec.activityName}' is used by more than one node or task.`);
    }
    names.add(spec.activityName);
  }
  return new LangGraphDefinitions<Graphs, Entrypoints>(internals);
}
