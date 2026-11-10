/** Native LangSmith lifecycle overrides, scoped to this plugin's workflow handlers. */

import { AsyncLocalStorage } from 'node:async_hooks';
import type { Client } from 'langsmith';
import type { KVMap } from 'langsmith/schemas';
import type { RunTree, RunTreeConfig } from 'langsmith/run_trees';
import { v7 as uuidv7 } from 'uuid';
import { getRandomStream, proxySinks, workflowInfo } from '@temporalio/workflow';
import { isUntraced, patchRunTree, serializeRun, suppressRun, type RunTreeOverrides } from './run-tree';
import { LANGSMITH_SINK_NAME, type LangSmithSinks } from './sinks';
import type { WorkflowLangSmithConfig } from './workflow-interceptors';

const LANGSMITH_RUN_TREE_STREAM = 'package-langsmith-tracing';

interface WorkflowRunTreeContext {
  config: WorkflowLangSmithConfig;
  readOnly: boolean;
}

// The bundle rewrites `node:async_hooks` to the AsyncLocalStorage the SDK injects into the sandbox.
const context = new AsyncLocalStorage<WorkflowRunTreeContext>();

export function workflowRunTreeContext<T>(
  config: WorkflowLangSmithConfig,
  fn: () => T,
  readOnly = false
): T {
  return context.run({ config, readOnly }, fn);
}

function runId(readOnly: boolean): string {
  const bytes = new Uint8Array(16);

  if (readOnly) {
    const { unsafe } = workflowInfo();
    return uuidv7({ msecs: unsafe.now(), random: unsafe.random.fillRandom(bytes) });
  }

  return uuidv7({ msecs: Date.now(), random: getRandomStream(LANGSMITH_RUN_TREE_STREAM).fill(bytes) });
}

function workerOrServerTime(): number {
  const { unsafe } = workflowInfo();
  return unsafe.isReplayingHistoryEvents ? Date.now() : unsafe.now();
}

// Every write inside a workflow is routed through a sink instead of this client.
const NOOP_CLIENT = {
  createRun: async (): Promise<void> => { },
  updateRun: async (): Promise<void> => { },
} as unknown as Client;

// LangSmith calls getDefaultConfig and _resolveAddressing from its constructor. Patching them also covers
// RunTree aliases imported before installation, including its own createChild.
/** Preserve native root/child selection, ordering, and replica inheritance. */
export const WORKFLOW_RUN_TREE_OVERRIDES: RunTreeOverrides = {
  statics: {
    getDefaultConfig: (original) =>
      function (this: typeof RunTree, ...args: unknown[]): RunTreeConfig {
        const defaults = original.apply(this, args);
        const active = context.getStore();
        if (!active) return defaults;
        return {
          ...defaults,
          id: runId(active.readOnly),
          start_time: workerOrServerTime(),
          replicas: active.config.replicas,
          // Keep traceable's awaits independent of the worker's destination env.
          tracingEnabled: true,
        };
      },
    getSharedClient: (original) =>
      function (this: typeof RunTree, ...args: unknown[]): Client {
        return context.getStore() ? NOOP_CLIENT : original.apply(this, args);
      },
  },
  methods: {
    _resolveAddressing: (original) =>
      function (this: RunTree, config: RunTreeConfig, ...args: unknown[]): void {
        const active = context.getStore();
        if (!active) return original.call(this, config, ...args);
        const selected = config.project_name || config.address ? config : { ...config, ...active.config.destination };
        original.call(this, selected, ...args);
        if (!selected.project_name && !selected.address) suppressRun(this);
        // A run with no parent and no propagated trace starts a new trace, which LangSmith's env gates.
        if (!config.parent_run && !config.trace_id && active.config.tracingEnabled === false) suppressRun(this);
      },
    createChild: (original) =>
      function (this: RunTree, ...args: unknown[]): RunTree {
        const child: RunTree = original.apply(this, args);
        if (context.getStore() && isUntraced(this)) suppressRun(child);
        return child;
      },
    end: (original) =>
      async function (this: RunTree, outputs?: KVMap, error?: string, endTime?: number, ...args: unknown[]): Promise<void> {
        if (context.getStore() && endTime === undefined) endTime = workerOrServerTime();
        return original.call(this, outputs, error, endTime, ...args);
      },
    postRun: (original) =>
      async function (this: RunTree, excludeChildRuns = true, ...args: unknown[]): Promise<void> {
        if (!context.getStore()) return original.call(this, excludeChildRuns, ...args);
        // Native traceable can defer asynchronous input processing until postRun.
        if ((this as RunTree & { _awaitInputsOnPost?: boolean })._awaitInputsOnPost) {
          this.inputs = await this.inputs;
        }
        if (!workflowInfo().unsafe.isReplayingHistoryEvents && !isUntraced(this)) {
          proxySinks<LangSmithSinks>()[LANGSMITH_SINK_NAME].createRun(serializeRun(this));
        }
        if (!excludeChildRuns) {
          for (const child of this.child_runs) await child.postRun(false);
        }
        this.child_runs = [];
      },
    patchRun: (original) =>
      async function (this: RunTree, options?: Parameters<RunTree['patchRun']>[0], ...args: unknown[]): Promise<void> {
        if (!context.getStore()) return original.call(this, options, ...args);
        if (!workflowInfo().unsafe.isReplayingHistoryEvents && !isUntraced(this)) {
          proxySinks<LangSmithSinks>()[LANGSMITH_SINK_NAME].updateRun(serializeRun(this), options);
        }
        this.child_runs = [];
      },
  },
};

let installed = false;

export function installRunTreeOverrides(): void {
  if (installed) return;
  patchRunTree(WORKFLOW_RUN_TREE_OVERRIDES);
  installed = true;
}
