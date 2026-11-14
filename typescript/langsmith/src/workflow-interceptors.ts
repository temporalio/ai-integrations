/**
 * Workflow-isolate interceptors and the LangSmith context provider. Loaded into
 * the workflow bundle (via the plugin's `workflowModules`) and runs entirely
 * inside the V8 isolate. Trace context propagates through the real,
 * async-context-tracking `AsyncLocalStorage` the SDK worker injects onto the
 * workflow-sandbox global; the plugin and LangSmith's own load-time init
 * converge on one shared store (see {@link sharedStore}).
 *
 * @module
 * @internal
 */

import { RunTree } from 'langsmith/run_trees';
import { AsyncLocalStorageProviderSingleton } from 'langsmith/singletons/traceable';
import type {
  ContinueAsNewInput,
  Next,
  WorkflowExecuteInput,
  WorkflowInboundCallsInterceptor,
  WorkflowInterceptors,
  WorkflowOutboundCallsInterceptor,
} from '@temporalio/workflow';
import {
  ContinueAsNew,
  isCancellation,
  workflowInfo,
  type ActivityInput,
  type Headers,
  type LocalActivityInput,
  type QueryInput,
  type SignalInput,
  type SignalWorkflowInput,
  type StartChildWorkflowExecutionInput,
  type StartNexusOperationInput,
  type StartNexusOperationOutput,
  type UpdateInput,
} from '@temporalio/workflow';
import {
  HEADER_KEY,
  encodeContextString,
  isInternalQuery,
  readContextHeader,
  withContextHeader,
  type LangSmithTraceContext,
} from './propagation';
import {
  RUN_TYPE,
  type Destination,
  type RunEnd,
  isUntraced,
  statusOutputs,
  suppressRun,
  thrownEnd,
  handleQueryRunName,
  handleSignalRunName,
  handleUpdateRunName,
  runHeaders,
  runTreeFromContext,
  runWorkflowRunName,
  signalChildWorkflowRunName,
  signalExternalWorkflowRunName,
  startActivityRunName,
  startChildWorkflowRunName,
  startNexusOperationRunName,
  validateUpdateRunName,
  wrapWithRun,
} from './run-tree';
import { destination, type ReplicaRouting } from './routing';
import { installRunTreeOverrides, workflowRunTreeContext } from './workflow-run-tree';

/**
 * Configuration the worker injects into the workflow bundle via the bundler's
 * `DefinePlugin`. Mirrors the user-facing plugin options that the workflow
 * isolate needs in order to decide whether to emit Temporal-operation runs.
 *
 * @internal
 */
export interface WorkflowLangSmithConfig {
  addTemporalRuns: boolean;
  replicas?: ReplicaRouting[];
  /**
   * The plugin's project or address, else LangSmith's env defaults, resolved outside the isolate when the bundle is
   * built. New roots use this destination; propagated parents retain their own. Absent when neither names one.
   */
  destination?: Destination;
  /** LangSmith's env tracing state outside the isolate. When false, only runs under a propagated parent are traced. */
  tracingEnabled?: boolean;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

// Injected at bundle time by the plugin's DefinePlugin. Declared so the bare
// identifier is replaced textually (avoids the dotted-access DefinePlugin
// pitfall). Absent in environments that bundled without the plugin.
declare const __TEMPORAL_LANGSMITH_CONFIG__: string | undefined;

function readConfig(): WorkflowLangSmithConfig {
  try {
    if (typeof __TEMPORAL_LANGSMITH_CONFIG__ === 'string') {
      return JSON.parse(__TEMPORAL_LANGSMITH_CONFIG__) as WorkflowLangSmithConfig;
    }
  } catch {
    /* fall through to default */
  }
  return { addTemporalRuns: false };
}

/** The `getStore`/`run` slice of `AsyncLocalStorage` the interceptors and LangSmith share. */
interface RunTreeStore {
  getStore(): RunTree | undefined;
  run<T>(store: RunTree | undefined, fn: () => T): T;
}

// The SDK worker injects a real, async-context-tracking `AsyncLocalStorage` onto
// the workflow-sandbox global before the bundle evaluates. Bundler aliases (see
// plugin.ts) also route LangSmith's `new AsyncLocalStorage()` there, so both the
// plugin and LangSmith's load-time `initializeGlobalInstance` compete for one
// global slot (first writer wins). Install our instance, then read back whichever
// won so interceptors and `traceable` read/write the same store regardless of
// load order.
declare const globalThis: { AsyncLocalStorage: new () => RunTreeStore };
AsyncLocalStorageProviderSingleton.initializeGlobalInstance(
  new globalThis.AsyncLocalStorage() as Parameters<
    typeof AsyncLocalStorageProviderSingleton.initializeGlobalInstance
  >[0]
);
const sharedStore = AsyncLocalStorageProviderSingleton.getInstance() as RunTreeStore;

/**
 * Reconstruct the propagated parent run from a Payload-keyed header map. A header naming no destination takes
 * `config.destination`, as LangSmith would take its env defaults outside the isolate. This plugin never propagates a
 * run with no destination (see {@link runHeaders}), so such a header comes from an older release; if the Worker has
 * no destination either, LangSmith fills in `"default"` for it here.
 */
function reconstructParent(
  headers: Headers | undefined,
  config: WorkflowLangSmithConfig,
  readOnly: boolean
): RunTree | undefined {
  const context = readContextHeader(headers);
  return workflowRunTreeContext(config, () => runTreeFromContext(context, config.destination), readOnly);
}

/** The `RunTree.end` arguments for a Workflow run whose handler threw `err`; continue-as-new isn't an error. */
function workflowThrownEnd(err: unknown): RunEnd {
  if (isCancellation(err)) return [statusOutputs('cancelled'), undefined];
  return err instanceof ContinueAsNew ? [statusOutputs('continued_as_new'), undefined] : thrownEnd(err);
}

/** Keep the incoming parent ambient when a Temporal run has no destination. */
function bodyAmbient(run: RunTree, parent: RunTree | undefined): RunTree {
  return isUntraced(run) ? parent ?? run : run;
}

/**
 * Emit a short-lived marker run as a child of `parent`. With no destination in `config` the marker has none either,
 * but it's built and awaited all the same: the isolate's awaits, and so the order of its commands, must not depend
 * on the Worker's env, or a history recorded under one env fails to replay under another.
 */
async function emitMarker(config: WorkflowLangSmithConfig, parent: RunTree, name: string): Promise<void> {
  const marker = parent.createChild({
    name,
    run_type: RUN_TYPE.CHAIN,
    tags: config.tags,
    extra: { metadata: { ...config.metadata } },
  });
  if (config.destination === undefined) suppressRun(marker);
  if (!marker.tracingEnabled) return;
  await marker.postRun();
  await marker.end(statusOutputs('ok'));
  await marker.patchRun();
}

function buildWorkflowRunTree(config: WorkflowLangSmithConfig, name: string, parent: RunTree | undefined): RunTree {
  const { workflowId, runId } = workflowInfo();
  const run = new RunTree({
    name,
    run_type: RUN_TYPE.CHAIN,
    parent_run: parent,
    parent_run_id: parent?.id,
    ...destination(parent, config.destination),
    replicas: parent?.replicas ?? config.replicas,
    tags: config.tags,
    extra: { metadata: { ...config.metadata, temporalWorkflowID: workflowId, temporalRunID: runId } },
  });
  if (config.destination === undefined) suppressRun(run);
  return run;
}

class LangSmithWorkflowInbound implements WorkflowInboundCallsInterceptor {
  constructor(private readonly config: WorkflowLangSmithConfig) { }

  async execute(input: WorkflowExecuteInput, next: Next<WorkflowInboundCallsInterceptor, 'execute'>): Promise<unknown> {
    return this.runInbound(runWorkflowRunName(workflowInfo().workflowType), input.headers, () => next(input));
  }

  async handleSignal(input: SignalInput, next: Next<WorkflowInboundCallsInterceptor, 'handleSignal'>): Promise<void> {
    await this.runInbound(handleSignalRunName(input.signalName), input.headers, () => next(input));
  }

  async handleQuery(input: QueryInput, next: Next<WorkflowInboundCallsInterceptor, 'handleQuery'>): Promise<unknown> {
    if (isInternalQuery(input.queryName)) {
      return next(input);
    }
    return this.runInbound(handleQueryRunName(input.queryName), input.headers, () => next(input), true);
  }

  async handleUpdate(
    input: UpdateInput,
    next: Next<WorkflowInboundCallsInterceptor, 'handleUpdate'>
  ): Promise<unknown> {
    return this.runInbound(handleUpdateRunName(input.name), input.headers, () => next(input));
  }

  validateUpdate(input: UpdateInput, next: Next<WorkflowInboundCallsInterceptor, 'validateUpdate'>): void {
    const parent = reconstructParent(input.headers, this.config, true);
    workflowRunTreeContext(this.config, () => {
      if (!this.config.addTemporalRuns) {
        sharedStore.run(parent, () => next(input));
        return;
      }
      const run = buildWorkflowRunTree(this.config, validateUpdateRunName(input.name), parent);
      void run.postRun();
      let end: RunEnd = [statusOutputs('ok'), undefined];
      try {
        sharedStore.run(bodyAmbient(run, parent), () => next(input));
      } catch (err) {
        end = workflowThrownEnd(err);
        throw err;
      } finally {
        void run.end(...end);
        void run.patchRun();
      }
    }, true);
  }

  /**
   * Open the inbound run, install it (see {@link bodyAmbient}) as the active run for the whole `next`
   * async scope, and close it. The real ALS scopes the run to this branch, so a
   * handler running concurrently with the workflow body never leaks its run into
   * the body's context.
   */
  private async runInbound(
    name: string,
    headers: Headers,
    next: () => Promise<unknown>,
    readOnly = false
  ): Promise<unknown> {
    const parent = reconstructParent(headers, this.config, readOnly);
    return workflowRunTreeContext(this.config, () => {
      if (!this.config.addTemporalRuns) {
        return sharedStore.run(parent, next);
      }
      const run = buildWorkflowRunTree(this.config, name, parent);
      void run.postRun();
      return sharedStore.run(bodyAmbient(run, parent), () => wrapWithRun(run, next, workflowThrownEnd));
    }, readOnly);
  }
}

class LangSmithWorkflowOutbound implements WorkflowOutboundCallsInterceptor {
  constructor(private readonly config: WorkflowLangSmithConfig) { }

  async scheduleActivity(
    input: ActivityInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'scheduleActivity'>
  ): Promise<unknown> {
    return this.withHeaders(startActivityRunName(input.activityType), input, next);
  }

  async scheduleLocalActivity(
    input: LocalActivityInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'scheduleLocalActivity'>
  ): Promise<unknown> {
    return this.withHeaders(startActivityRunName(input.activityType), input, next);
  }

  async startChildWorkflowExecution(
    input: StartChildWorkflowExecutionInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'startChildWorkflowExecution'>
  ): Promise<[Promise<string>, Promise<unknown>]> {
    return this.withHeaders(startChildWorkflowRunName(input.workflowType), input, next);
  }

  continueAsNew(
    input: ContinueAsNewInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'continueAsNew'>
  ): Promise<never> {
    return next({ ...input, headers: withContextHeader(input.headers, runHeaders(sharedStore.getStore())) });
  }

  async signalWorkflow(
    input: SignalWorkflowInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'signalWorkflow'>
  ): Promise<void> {
    const name = input.target.type === 'child'
      ? signalChildWorkflowRunName(input.signalName)
      : signalExternalWorkflowRunName(input.signalName);
    return this.withHeaders(name, input, next);
  }

  async startNexusOperation(
    input: StartNexusOperationInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'startNexusOperation'>
  ): Promise<StartNexusOperationOutput> {
    return this.traceOutbound(startNexusOperationRunName(input.service, input.operation), (context) => {
      const headers = context ? { ...input.headers, [HEADER_KEY]: encodeContextString(context) } : input.headers;
      return next({ ...input, headers });
    });
  }

  /** Emit the marker, then pass `call` the caller's context, so the remote run is the marker's sibling. */
  private async traceOutbound<R>(
    name: string,
    call: (context: LangSmithTraceContext | undefined) => Promise<R>
  ): Promise<R> {
    const ambient = sharedStore.getStore();
    if (this.config.addTemporalRuns && ambient !== undefined) {
      await emitMarker(this.config, ambient, name);
    }
    return call(runHeaders(ambient));
  }

  private withHeaders<I extends { headers: Headers }, R>(
    name: string,
    input: I,
    next: (input: I) => Promise<R>
  ): Promise<R> {
    return this.traceOutbound(name, (context) =>
      next({ ...input, headers: withContextHeader(input.headers, context) })
    );
  }
}

/**
 * Workflow interceptors factory. Loaded by the Temporal worker for every
 * workflow in the bundle (registered via the plugin's `workflowModules`).
 *
 * @internal
 */
export function interceptors(): WorkflowInterceptors {
  const config = readConfig();
  installRunTreeOverrides();
  return {
    inbound: [new LangSmithWorkflowInbound(config)],
    outbound: [new LangSmithWorkflowOutbound(config)],
  };
}
