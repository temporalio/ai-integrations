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
  scrubSensitive,
  withContextHeader,
} from './propagation';
import {
  RUN_TYPE,
  type Destination,
  asOutputs,
  describeError,
  emitMarkerRun,
  isUntraced,
  suppressRun,
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
  headers: Record<string, unknown> | undefined,
  config: WorkflowLangSmithConfig,
  generateNewUuid4?: () => string
): RunTree | undefined {
  const context = readContextHeader(headers as Record<string, never> | undefined);
  return workflowRunTreeContext(config, () => runTreeFromContext(context, config.destination), generateNewUuid4);
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
async function emitMarker(
  config: WorkflowLangSmithConfig,
  parent: RunTree,
  name: string,
  inputs: Record<string, unknown>
): Promise<RunTree> {
  const marker = parent.createChild({ name, run_type: RUN_TYPE.CHAIN, inputs });
  if (config.destination === undefined) suppressRun(marker);
  await emitMarkerRun(marker);
  return marker;
}

function buildWorkflowRunTree(
  config: WorkflowLangSmithConfig,
  params: {
    name: string;
    runType: string;
    parent: RunTree | undefined;
    inputs: Record<string, unknown>;
  }
): RunTree {
  const run = new RunTree({
    name: params.name,
    run_type: params.runType,
    parent_run: params.parent,
    parent_run_id: params.parent?.id,
    ...destination(params.parent, config.destination),
    replicas: params.parent?.replicas ?? config.replicas,
    tags: config.tags,
    extra: { metadata: scrubSensitive(config.metadata) ?? {} },
    inputs: params.inputs,
  });
  if (config.destination === undefined) suppressRun(run);
  return run;
}

class LangSmithWorkflowInbound implements WorkflowInboundCallsInterceptor {
  constructor(private readonly config: WorkflowLangSmithConfig) {}

  async execute(input: WorkflowExecuteInput, next: Next<WorkflowInboundCallsInterceptor, 'execute'>): Promise<unknown> {
    const parent = reconstructParent(input.headers, this.config);
    return this.runInbound(
      runWorkflowRunName(workflowInfo().workflowType),
      RUN_TYPE.CHAIN,
      parent,
      { args: input.args },
      () => next(input)
    );
  }

  async handleSignal(input: SignalInput, next: Next<WorkflowInboundCallsInterceptor, 'handleSignal'>): Promise<void> {
    const parent = reconstructParent(input.headers, this.config);
    await this.runInbound(handleSignalRunName(input.signalName), RUN_TYPE.CHAIN, parent, { args: input.args }, () =>
      next(input)
    );
  }

  async handleQuery(input: QueryInput, next: Next<WorkflowInboundCallsInterceptor, 'handleQuery'>): Promise<unknown> {
    if (isInternalQuery(input.queryName)) {
      return next(input);
    }
    const parent = reconstructParent(input.headers, this.config, workflowInfo().unsafe.random.uuid4);
    return this.runInbound(
      handleQueryRunName(input.queryName),
      RUN_TYPE.CHAIN,
      parent,
      { args: input.args },
      () => next(input),
      // Read-only: mint run ids from the nondeterministic unsafe random source so the
      // handler never draws from the Workflow main PRNG (which a clean replay would not advance).
      workflowInfo().unsafe.random.uuid4
    );
  }

  async handleUpdate(
    input: UpdateInput,
    next: Next<WorkflowInboundCallsInterceptor, 'handleUpdate'>
  ): Promise<unknown> {
    const parent = reconstructParent(input.headers, this.config);
    return this.runInbound(handleUpdateRunName(input.name), RUN_TYPE.CHAIN, parent, { args: input.args }, () =>
      next(input)
    );
  }

  validateUpdate(input: UpdateInput, next: Next<WorkflowInboundCallsInterceptor, 'validateUpdate'>): void {
    // Read-only: mint run ids from the nondeterministic unsafe random source so the validator
    // never draws from the Workflow main PRNG (which a clean replay would not advance).
    const generateNewUuid4 = workflowInfo().unsafe.random.uuid4;
    workflowRunTreeContext(this.config, () => {
      const parent = reconstructParent(input.headers, this.config);
      if (!this.config.addTemporalRuns) {
        sharedStore.run(parent, () => next(input));
        return;
      }
      const run = buildWorkflowRunTree(this.config, {
        name: validateUpdateRunName(input.name),
        runType: RUN_TYPE.CHAIN,
        parent,
        inputs: { args: input.args },
      });
      void run.postRun();
      try {
        sharedStore.run(bodyAmbient(run, parent), () => next(input));
        void run.end({});
      } catch (err) {
        void run.end(undefined, describeError(err));
        void run.patchRun();
        throw err;
      }
      void run.patchRun();
    }, generateNewUuid4);
  }

  /**
   * Open the inbound run, install it (see {@link bodyAmbient}) as the active run for the whole `next`
   * async scope, and close it. The real ALS scopes the run to this branch, so a
   * handler running concurrently with the workflow body never leaks its run into
   * the body's context.
   */
  private async runInbound(
    name: string,
    runType: string,
    parent: RunTree | undefined,
    inputs: Record<string, unknown>,
    next: () => Promise<unknown>,
    generateNewUuid4?: () => string
  ): Promise<unknown> {
    return workflowRunTreeContext(this.config, () => {
      if (!this.config.addTemporalRuns) {
        return sharedStore.run(parent, next);
      }
      const run = buildWorkflowRunTree(this.config, { name, runType, parent, inputs });
      void run.postRun();
      const body = async (): Promise<unknown> => {
        try {
          const result = await next();
          await run.end(asOutputs(result));
          await run.patchRun();
          return result;
        } catch (err) {
          await run.end(undefined, describeError(err));
          await run.patchRun();
          throw err;
        }
      };
      return sharedStore.run(bodyAmbient(run, parent), body);
    }, generateNewUuid4);
  }
}

class LangSmithWorkflowOutbound implements WorkflowOutboundCallsInterceptor {
  constructor(private readonly config: WorkflowLangSmithConfig) {}

  async scheduleActivity(
    input: ActivityInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'scheduleActivity'>
  ): Promise<unknown> {
    return this.peerMarker(startActivityRunName(input.activityType), input, (headers: Headers) =>
      next({ ...input, headers })
    );
  }

  async scheduleLocalActivity(
    input: LocalActivityInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'scheduleLocalActivity'>
  ): Promise<unknown> {
    return this.peerMarker(startActivityRunName(input.activityType), input, (headers: Headers) =>
      next({ ...input, headers })
    );
  }

  async startChildWorkflowExecution(
    input: StartChildWorkflowExecutionInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'startChildWorkflowExecution'>
  ): Promise<[Promise<string>, Promise<unknown>]> {
    return this.peerMarker(startChildWorkflowRunName(input.workflowType), input, (headers: Headers) =>
      next({ ...input, headers })
    );
  }

  continueAsNew(
    input: ContinueAsNewInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'continueAsNew'>
  ): Promise<never> {
    const context = runHeaders(sharedStore.getStore());
    const headers = withContextHeader(input.headers, context);
    return next({ ...input, headers });
  }

  async signalWorkflow(
    input: SignalWorkflowInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'signalWorkflow'>
  ): Promise<void> {
    const isChild = input.target.type === 'child';
    const name = isChild
      ? signalChildWorkflowRunName(input.signalName)
      : signalExternalWorkflowRunName(input.signalName);
    return this.parentMarker(name, input, (headers: Headers) => next({ ...input, headers }));
  }

  async startNexusOperation(
    input: StartNexusOperationInput,
    next: Next<WorkflowOutboundCallsInterceptor, 'startNexusOperation'>
  ): Promise<StartNexusOperationOutput> {
    const ambient = sharedStore.getStore();
    if (this.config.addTemporalRuns && ambient !== undefined) {
      await emitMarker(this.config, ambient, startNexusOperationRunName(input.service, input.operation), {});
    }
    const ctx = runHeaders(ambient);
    const headers = ctx ? { ...input.headers, [HEADER_KEY]: encodeContextString(ctx) } : input.headers;
    return next({ ...input, headers });
  }

  /** Sibling-marker operations: marker is a peer of the remote run; propagate ambient. */
  private async peerMarker<R>(
    name: string,
    input: { headers: Headers; args?: unknown[] },
    next: (headers: Headers) => Promise<R>
  ): Promise<R> {
    const ambient = sharedStore.getStore();
    if (this.config.addTemporalRuns && ambient !== undefined) {
      await emitMarker(this.config, ambient, name, { args: input.args ?? [] });
    }
    const headers = withContextHeader(input.headers, runHeaders(ambient));
    return next(headers);
  }

  /** Parent-marker operations: remote handler nests under the marker; propagate marker. */
  private async parentMarker<R>(
    name: string,
    input: { headers: Headers; args?: unknown[] },
    next: (headers: Headers) => Promise<R>
  ): Promise<R> {
    const ambient = sharedStore.getStore();
    let propagate: RunTree | undefined = ambient;
    if (this.config.addTemporalRuns && ambient !== undefined) {
      const marker = await emitMarker(this.config, ambient, name, { args: input.args ?? [] });
      // A marker with no destination is never propagated, so the remote handler nests under the ambient run instead.
      propagate = isUntraced(marker) ? ambient : marker;
    }
    const headers = withContextHeader(input.headers, runHeaders(propagate));
    return next(headers);
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
