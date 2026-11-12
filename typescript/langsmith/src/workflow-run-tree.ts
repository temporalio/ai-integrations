/** Native LangSmith lifecycle overrides, scoped to this plugin's workflow handlers. */

import type { Client } from 'langsmith';
import { RunTree, type RunTreeConfig } from 'langsmith/run_trees';
import { getRandomStream, proxySinks, workflowInfo } from '@temporalio/workflow';
import { isUntraced, serializeRun, suppressRun } from './run-tree';
import { LANGSMITH_SINK_NAME, type LangSmithSinks } from './sinks';
import type { WorkflowLangSmithConfig } from './workflow-interceptors';

// Keep the existing stream name stable for replay compatibility.
const LANGSMITH_RUN_TREE_STREAM = 'package-langsmith-tracing';

interface WorkflowRunTreeContext {
  config: WorkflowLangSmithConfig;
  generateNewUuid4?: () => string;
}

interface ContextStore {
  getStore(): WorkflowRunTreeContext | undefined;
  run<T>(context: WorkflowRunTreeContext, fn: () => T): T;
}

declare const globalThis: { AsyncLocalStorage: new () => ContextStore };
const context = new globalThis.AsyncLocalStorage();

export function workflowRunTreeContext<T>(
  config: WorkflowLangSmithConfig,
  fn: () => T,
  generateNewUuid4: (() => string) | undefined = context.getStore()?.generateNewUuid4
): T {
  return context.run({ config, generateNewUuid4 }, fn);
}

// Every write inside a workflow is routed through a sink instead of this client.
const NOOP_CLIENT = {
  createRun: async (): Promise<void> => {},
  updateRun: async (): Promise<void> => {},
} as unknown as Client;

// LangSmith calls these hooks from its constructor. Patching them also covers
// RunTree aliases imported before installation, including its own createChild.
interface ConstructorHooks {
  getDefaultConfig(): RunTreeConfig;
}
interface AddressingHook {
  _resolveAddressing(config: RunTreeConfig): void;
}

let installed = false;

/** Preserve native root/child selection, ordering, and replica inheritance. */
export function installRunTreeOverrides(): void {
  if (installed) return;
  installed = true;
  const constructor = RunTree as unknown as ConstructorHooks;
  const prototype = RunTree.prototype as unknown as AddressingHook;
  const originalDefaults = constructor.getDefaultConfig;
  const originalAddressing = prototype._resolveAddressing;
  const originalClient = RunTree.getSharedClient;
  const originalChild = RunTree.prototype.createChild;
  const originalPost = RunTree.prototype.postRun;
  const originalPatch = RunTree.prototype.patchRun;

  constructor.getDefaultConfig = function () {
    const defaults = originalDefaults.call(this);
    const active = context.getStore();
    if (!active) return defaults;
    return {
      ...defaults,
      id: active.generateNewUuid4?.() ?? getRandomStream(LANGSMITH_RUN_TREE_STREAM).uuid4(),
      replicas: active.config.replicas,
      // Keep traceable's awaits independent of the worker's destination env.
      tracingEnabled: true,
    };
  };
  RunTree.getSharedClient = function () {
    return context.getStore() ? NOOP_CLIENT : originalClient.call(this);
  };
  prototype._resolveAddressing = function (this: RunTree, config) {
    const active = context.getStore();
    if (!active) return originalAddressing.call(this, config);
    const selected = config.project_name || config.address ? config : { ...config, ...active.config.destination };
    originalAddressing.call(this, selected);
    if (!selected.project_name && !selected.address) suppressRun(this);
  };
  RunTree.prototype.createChild = function (config) {
    const child = originalChild.call(this, config);
    if (context.getStore() && isUntraced(this)) suppressRun(child);
    return child;
  };
  RunTree.prototype.postRun = async function (excludeChildRuns = true) {
    if (!context.getStore()) return originalPost.call(this, excludeChildRuns);
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
  };
  RunTree.prototype.patchRun = async function (options) {
    if (!context.getStore()) return originalPatch.call(this, options);
    if (!workflowInfo().unsafe.isReplayingHistoryEvents && !isUntraced(this)) {
      proxySinks<LangSmithSinks>()[LANGSMITH_SINK_NAME].updateRun(serializeRun(this), options);
    }
    this.child_runs = [];
  };
}
