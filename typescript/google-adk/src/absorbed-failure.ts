/**
 * Surfaces the failures `@google/adk` absorbs before Workflow code can see them.
 * `LlmAgent.runAndHandleError` turns any `Error` escaping a model call into an event
 * and lets the run finish, so a Workflow whose model Activity failed returns
 * normally. `TemporalModel` records such a failure here, and the inbound interceptor
 * below re-raises it once the Workflow or handler frame that recorded it returns.
 * ADK absorbs a cancellation outside a model call too, in a tool or a model callback;
 * that one is recorded where ADK hands the error to its plugin manager (see
 * {@link watchErrorHandoffs}).
 *
 * Await an agent turn in the frame that started it: a failure recorded while that frame
 * is open surfaces there, and one recorded after a handler frame returned surfaces in the
 * main function's. One recorded after the main function returned never surfaces —
 * completing with unfinished handlers is a user error Temporal already warns about. A
 * cancellation a Signal handler absorbed is the exception: it surfaces in the main
 * function's frame too, so the main function's cleanup runs before the execution ends.
 * Only the execution's own cancellation is recorded at all (see {@link executionCancellation}).
 *
 * The same interceptor also converts the errors ADK 2.0's workflow runtime *throws* —
 * a node past its `timeout`, an output that failed its schema, an approval ADK refused
 * to bind — from the plain `Error`s they are into non-retryable `ApplicationFailure`s.
 * The Temporal SDK fails a Workflow only for a `TemporalFailure` (or a configured
 * failure type); any other error fails the Workflow *Task*, which the server retries
 * forever. These are expected runtime outcomes of a graph, not bugs, so they must end
 * the execution (or reject the Update) with a typed failure instead.
 */

import type { AsyncLocalStorage as ALS } from 'node:async_hooks';

import { BaseNode, FunctionTool, PluginManager } from '@google/adk';
import { ApplicationFailure, TemporalFailure } from '@temporalio/common';
import {
  AsyncLocalStorage,
  CancellationScope,
  ContinueAsNew,
  inWorkflowContext,
  isCancellation,
  type WorkflowInterceptorsFactory,
} from '@temporalio/workflow';

import { ADK_RUNTIME_FAILURE_TYPES } from './error-types';

// Held on the per-execution sandbox `globalThis` rather than in module scope: a
// bundle can hold two copies of this module (the interceptor list registers its
// compiled path, while a Workflow reaches it through its own import of
// `TemporalModel`), and both have to reach the same recordings.
const ABSORBED = '__temporal_googleAdkAbsorbedFailures';

/**
 * One `TemporalModel` call made by an ADK agent inside a Workflow.
 *
 * @internal
 */
export interface ModelCall {
  /** The `adk_agent_name` label of the request. */
  agent: string;
  /**
   * The ADK invocation the call belongs to, identified by the `AbortSignal` ADK hands the
   * model (`InvocationContext.abortSignal`). A `Workflow` run makes one signal and gives
   * every node the same object, so a node retry and a re-activation of the same node share
   * it while a later turn does not. `undefined` for a plain agent turn, where the runner
   * sets no signal and one turn cannot be told from the next.
   */
  invocation: AbortSignal | undefined;
  /**
   * The call's number among the model calls this execution has started, taken when the
   * call starts rather than when it settles. Workflow code starts its calls in the same
   * order on every replay, so the number is deterministic.
   */
  seq: number;
}

/** A model call that failed and that ADK absorbed into an event instead of rethrowing. */
interface Recording extends ModelCall {
  /** The failure the `TemporalModel` call threw. */
  error: unknown;
}

/** What one inbound frame — the main function, a Signal handler, an Update handler — absorbed. */
interface Frame {
  /** Failures no caller has marked handled or superseded, in the order they were absorbed. */
  pending: Recording[];
  /**
   * The frame's first absorbed cancellation of the execution, held apart from `pending`: it
   * outranks anything else the frame absorbed.
   */
  cancellation?: unknown;
  /** Whether the frame has returned, spending its one chance to raise; it never takes another. */
  surfaced: boolean;
}

interface AbsorbedFailures {
  /** The frame the running code belongs to. */
  frames: ALS<Frame>;
  /** The main function's frame, which owns whatever is absorbed outside a handler frame. */
  main?: Frame;
  /** How many model calls the execution has started: the latest call's `seq`. */
  calls: number;
  /**
   * The `seq` of each agent's latest answered call, per ADK invocation. Held apart from the
   * frames, because a call can settle after the frame that started it has returned.
   */
  answered: WeakMap<AbortSignal, Map<string, number>>;
  /**
   * The reason the root scope was cancelled with, once the execution's cancellation request
   * cancelled it. The SDK hands a scope's reason on to the scopes it cancels as the same
   * object, so an operation that this request cancelled sees exactly this reason.
   */
  executionCancel?: unknown;
  /** The failures of operations whose scope this request cancelled first (see {@link watchScope}). */
  cancelledByExecution: WeakSet<object>;
  /** The ADK node attempt the running code belongs to (see {@link watchNodeAttempts}). */
  attempts: ALS<NodeAttempt>;
  /** What a tool or a plugin hook threw, per the plain `Error` ADK threw in its place (see {@link keepWrapped}). */
  wrapped: WeakMap<object, unknown>;
  /** Where the running tool call keeps what its `execute` throws (see {@link watchWrappedErrors}). */
  toolCall: ALS<Thrown>;
}

function recorded(): AbsorbedFailures {
  const global = globalThis as Record<string, unknown>;
  let state = global[ABSORBED] as AbsorbedFailures | undefined;
  if (state === undefined) {
    state = {
      frames: new AsyncLocalStorage<Frame>(),
      calls: 0,
      answered: new WeakMap(),
      cancelledByExecution: new WeakSet(),
      attempts: new AsyncLocalStorage<NodeAttempt>(),
      wrapped: new WeakMap(),
      toolCall: new AsyncLocalStorage<Thrown>(),
    };
    global[ABSORBED] = state;
  }
  return state;
}

function openFrame(): Frame | undefined {
  const { frames, main } = recorded();
  const frame = frames.getStore() ?? main;
  if (frame?.surfaced === false) return frame;
  // A surfaced frame is never read again; only the main function's can still raise a failure.
  return main?.surfaced === false ? main : undefined;
}

/**
 * Numbers a `TemporalModel` call as it starts, so that how it ends can be ordered against
 * the same agent's other calls (see {@link recordModelSuccess}).
 *
 * @internal
 */
export function startModelCall(agent: string, invocation: AbortSignal | undefined): ModelCall {
  const state = recorded();
  state.calls += 1;
  return { agent, invocation, seq: state.calls };
}

/**
 * Whether the same agent has answered in the same invocation through a call that started
 * after `call`. Then `call` belongs to an attempt ADK has already moved past: a call is
 * left running that way by ADK's own node `timeout`, which abandons the timed-out attempt
 * without cancelling its model call (`node_runner.ts`) while a retry answers.
 *
 * The key is the one {@link recordModelSuccess} spends by, so one agent run side by side
 * over several items (a `ParallelWorker`) is one agent here: an item that answered hides
 * the failure of an item whose call started earlier, and the run then fails on ADK's own
 * report of that item (`NodeReportedError`) rather than on the model failure.
 */
function superseded(call: ModelCall): boolean {
  if (call.invocation === undefined) return false;
  const answered = recorded().answered.get(call.invocation)?.get(call.agent);
  return answered !== undefined && answered > call.seq;
}

/**
 * Records how a failed `TemporalModel` call ended, unless the failure is one ADK has
 * already moved past ({@link superseded}): a call left running by a timed-out attempt
 * whose retry answered first says nothing about the run, however it ends later. A
 * cancellation is decided by the call's own node attempt instead ({@link recordCancellation}).
 *
 * @internal
 */
export function recordAbsorbedFailure(err: unknown, call: ModelCall): void {
  if (isCancellation(err)) {
    recordCancellation(err);
    return;
  }
  if (superseded(call)) return;
  openFrame()?.pending.push({ ...call, error: err });
}

/**
 * Watches the root scope for the execution's cancellation request and keeps the reason it
 * cancels the root scope with. Run once per execution, before Workflow code starts, so this
 * reaction runs before any scope the request cancels in turn can react.
 */
function watchExecutionCancel(): void {
  const state = recorded();
  let root = CancellationScope.current();
  while (root.parent !== undefined) root = root.parent;
  root.cancelRequested.catch((reason: unknown) => {
    state.executionCancel = reason;
  });
}

/**
 * Marks how the operations started in the current scope fail when the execution's
 * cancellation request was what cancelled them: the first cancellation that reached that
 * scope came from the root scope. Whatever scope later handles the failure (one that the
 * operation's own code has left, ADK's hand-off) cannot tell that any more: an inner scope
 * (`CancellationScope.withTimeout`) may have cancelled the operation inside a
 * non-cancellable one while the root was cancelled too. A timer, a `condition` or an
 * operation started in an already cancelled scope fails with the scope's reason itself and
 * needs no mark. Returns the function that watches each such operation.
 */
function watchScope(): (operation: Promise<unknown>) => void {
  const scope = CancellationScope.current();
  if (!scope.cancellable) return () => undefined;
  let reason: unknown;
  scope.cancelRequested.catch((cancelled: unknown) => {
    reason = cancelled;
  });
  return (operation) => {
    operation.then(undefined, (err: unknown) => {
      const state = recorded();
      if (reason === undefined || reason !== state.executionCancel || !isCancellation(err)) return;
      if (typeof err === 'object' && err !== null) state.cancelledByExecution.add(err);
    });
  };
}

/** Returns `operation`, watched as {@link watchScope} describes. */
function watched<T>(operation: Promise<T>): Promise<T> {
  watchScope()(operation);
  return operation;
}

/**
 * Returns `started`, watched as {@link watchScope} describes, and so are the operations its
 * result holds: a Nexus operation cancelled before it starts fails `started` itself.
 */
function watchedEach<T>(started: Promise<T>, operations: (result: T) => Promise<unknown>[]): Promise<T> {
  const watch = watchScope();
  watch(started);
  started.then(
    (result) => operations(result).forEach(watch),
    () => undefined
  );
  return started;
}

/**
 * The failure through which the execution's cancellation request ended `err`: the root
 * scope's reason or the failure of an operation that request cancelled (see
 * {@link watchScope}), found on `err` itself, its `cause` chain, or what a tool or a plugin
 * hook threw before ADK wrapped it (see {@link keepWrapped}). `undefined` when nothing
 * there came from the request: an inner scope's cancel, or any other error, even one thrown
 * after the Workflow caught its cancellation.
 */
function executionCancellation(err: unknown): unknown {
  const state = recorded();
  for (const current of causes(err)) {
    if (current === state.executionCancel || state.cancelledByExecution.has(current)) return current;
  }
  return undefined;
}

/** `err`, its `cause` chain, and what a tool or a plugin hook threw before ADK wrapped it. */
function* causes(err: unknown): Generator<object> {
  const { wrapped } = recorded();
  const seen = new Set<unknown>();
  let current = err;
  while (typeof current === 'object' && current !== null && !seen.has(current)) {
    yield current;
    seen.add(current);
    current = wrapped.get(current) ?? (current as { cause?: unknown }).cause;
  }
}

/**
 * Records a cancellation ADK absorbed as the open frame's, but only when it came from the
 * execution's own cancellation request ({@link executionCancellation}). One that only an
 * inner scope caused is left out, whether it came before or after that request: the
 * Workflow may catch the request and finish on purpose, and an inner scope's cancel must
 * not end that execution CANCELLED. So is one absorbed in a node attempt ADK abandoned
 * ({@link abandoned}): ADK had moved past that work, and the run's outcome is the retry's.
 */
function recordCancellation(err: unknown): void {
  if (abandoned(recorded().attempts.getStore())) return;
  const cancellation = executionCancellation(err);
  if (cancellation === undefined) return;
  const frame = openFrame();
  if (frame !== undefined) frame.cancellation ??= cancellation;
}

/** One run of an ADK node: one attempt, as ADK's node runner retries a node. */
interface NodeAttempt {
  /** The attempt of the node that ran this one, if any. */
  parent: NodeAttempt | undefined;
  /** Whether ADK abandoned the attempt: its node's `timeout` fired while it ran. */
  abandoned: boolean;
}

/** Whether `attempt` or an attempt it ran in was abandoned (see {@link watchNodeAttempts}). */
function abandoned(attempt: NodeAttempt | undefined): boolean {
  for (let current = attempt; current !== undefined; current = current.parent) {
    if (current.abandoned) return true;
  }
  return false;
}

/** A `NodeContext`, as far as it is read here. */
interface AttemptContext {
  abortSignal?: AbortSignal;
  invocationContext?: { abortSignal?: AbortSignal };
}

/**
 * Runs each ADK node attempt in its own {@link NodeAttempt}, so a cancellation absorbed in
 * it can tell whether ADK has moved past it. ADK's node runner calls `BaseNode.run` once per
 * attempt, with the attempt's own `AbortSignal` on the `NodeContext` when the node has a
 * `timeout` (`node_runner.ts`, ADK 2.0 to 2.3). When the timeout fires, the runner aborts
 * that signal, gives up on the attempt without cancelling what it waits on (a model call, a
 * tool, a model callback), and retries the node or fails it. The runner also aborts it when
 * the invocation's own signal aborts; that is not ADK moving past one attempt, so it is not
 * counted. A retry, and another item of a `ParallelWorker`, runs as an attempt of its own,
 * which leaves this one's state alone. What ADK does is unchanged.
 */
function watchNodeAttempts(): void {
  const proto = BaseNode.prototype as unknown as Record<PropertyKey, unknown>;
  const run = proto.run as (this: unknown, ctx: AttemptContext, input: unknown) => AsyncGenerator<unknown>;
  proto.run = function (this: unknown, ctx: AttemptContext, input: unknown): AsyncIterableIterator<unknown> {
    const { attempts } = recorded();
    const attempt: NodeAttempt = { parent: attempts.getStore(), abandoned: false };
    const own = ctx?.abortSignal;
    const invocation = ctx?.invocationContext?.abortSignal;
    if (own !== undefined && own !== invocation && !own.aborted) {
      own.addEventListener(
        'abort',
        () => {
          attempt.abandoned = invocation?.aborted !== true;
        },
        { once: true }
      );
    }
    const iterator = run.call(this, ctx, input);
    // Every step of the run happens in the attempt, whoever drives it.
    return {
      next: (...args) => attempts.run(attempt, () => iterator.next(...args)),
      return: (value) => attempts.run(attempt, () => iterator.return(value)),
      throw: (error) => attempts.run(attempt, () => iterator.throw(error)),
      [Symbol.asyncIterator]() {
        return this;
      },
    };
  };
}

/** Marks the `PluginManager` prototype once its error hand-offs record cancellations. */
const WATCHED = Symbol.for('__temporal_googleAdkWatchedErrorHandoffs');

/** Told what code that ADK runs threw, for the plain `Error` ADK throws in its place. */
type Thrown = (error: unknown) => void;

/**
 * Runs `call`, an ADK method that throws a plain `Error` in place of the error that code it
 * runs threw, with no `cause`: `FunctionTool.runAsync` for a tool's `execute`, and
 * `PluginManager.runCallbacks` for a plugin hook before ADK 2.2.1. That error is kept for
 * the one ADK throws, so {@link executionCancellation} can follow it.
 */
function keepWrapped(call: (thrown: Thrown) => Promise<unknown>): Promise<unknown> {
  let thrown: { error: unknown } | undefined;
  const result = call((error) => {
    thrown = { error };
  });
  result.then(undefined, (wrapper: unknown) => {
    if (thrown !== undefined && thrown.error !== wrapper && typeof wrapper === 'object' && wrapper !== null) {
      recorded().wrapped.set(wrapper, thrown.error);
    }
  });
  return result;
}

/** `fn`, telling what it throws to the {@link Thrown} that `tell` returns when it is called. */
function telling<F extends (...args: never[]) => unknown>(fn: F, tell: () => Thrown | undefined): F {
  return function (this: unknown, ...args: Parameters<F>): unknown {
    const thrown = tell();
    try {
      const result = fn.apply(this, args);
      if (result instanceof Promise) result.then(undefined, (error: unknown) => thrown?.(error));
      return result;
    } catch (error) {
      thrown?.(error);
      throw error;
    }
  } as F;
}

type Execute = (...args: never[]) => unknown;

/** The `execute` functions {@link watchWrappedErrors} put in place of a tool's own. */
const TELLING_EXECUTES = new WeakSet<Execute>();

/**
 * Keeps the errors that a tool's `execute` and a plugin hook throw for the plain `Error`s
 * ADK throws in their place (see {@link keepWrapped}). A tool's `execute` is replaced, the
 * first time ADK runs the tool, by one that tells the call it runs in what it threw. What
 * ADK does is unchanged.
 */
function watchWrappedErrors(): void {
  const tool = FunctionTool.prototype as unknown as Record<PropertyKey, unknown>;
  const runAsync = tool.runAsync as (this: { execute: Execute }, request: unknown) => Promise<unknown>;
  tool.runAsync = function (this: { execute: Execute }, request: unknown): Promise<unknown> {
    if (!TELLING_EXECUTES.has(this.execute)) {
      this.execute = telling(this.execute, () => recorded().toolCall.getStore());
      TELLING_EXECUTES.add(this.execute);
    }
    return keepWrapped((thrown) => recorded().toolCall.run(thrown, () => runAsync.call(this, request)));
  };
  const manager = PluginManager.prototype as unknown as Record<PropertyKey, unknown>;
  const runCallbacks = manager.runCallbacks as (
    this: PluginManager,
    plugins: unknown,
    callback: (plugin: unknown) => unknown,
    name: string
  ) => Promise<unknown>;
  manager.runCallbacks = function (
    this: PluginManager,
    registered: unknown,
    callback: (plugin: unknown) => unknown,
    name: string
  ): Promise<unknown> {
    return keepWrapped((thrown) =>
      runCallbacks.call(
        this,
        registered,
        telling(callback, () => thrown),
        name
      )
    );
  };
}

/**
 * Wraps the two `PluginManager` methods ADK calls with an error it is about to absorb, so
 * that a cancellation it came from is recorded first. ADK hands the error a tool threw
 * (`functions.ts`), and the error of a model step, model callbacks included
 * (`LlmAgent.runAndHandleError`), to its plugin manager and then goes on: ADK before 2.3
 * drops a long-running tool's error, and a model callback's error becomes an error event
 * that ends the turn. What ADK does next is unchanged. Done once per Workflow, whichever
 * copy of this module loads first, together with {@link watchNodeAttempts}.
 */
function watchErrorHandoffs(): void {
  const proto = PluginManager.prototype as unknown as Record<PropertyKey, unknown>;
  if (proto[WATCHED] === true) return;
  proto[WATCHED] = true;
  watchWrappedErrors();
  watchNodeAttempts();
  for (const method of ['runOnToolErrorCallback', 'runOnModelErrorCallback']) {
    const original = proto[method] as (this: PluginManager, params: { error?: unknown }) => Promise<unknown>;
    proto[method] = function (this: PluginManager, params: { error?: unknown }): Promise<unknown> {
      recordCancellation(params.error);
      return original.call(this, params);
    };
  }
}

if (inWorkflowContext()) watchErrorHandoffs();

/**
 * Declares that a `TemporalModel` call succeeded, which spends whatever the same agent
 * absorbed in the same ADK invocation through a call that started before this one: that is
 * ADK having retried the node (or activated it again) and got its answer, and a run
 * finishing normally must not fail on the attempt it recovered from. Ordering by when the
 * calls started, not by when they ended, also covers an earlier call still running when
 * the answer comes: its failure is not recorded at all, whenever it arrives.
 *
 * Both halves of the key matter. A sibling agent answering says nothing about this one's
 * failure, and a later *turn* by the same agent is a new question, not a second go at the
 * one that failed — which is why a call with no invocation to compare (a plain agent turn
 * outside a graph, where ADK sets no signal) never clears anything. Nor does an answer
 * spend the failure of a call that started after it.
 *
 * @internal
 */
export function recordModelSuccess(call: ModelCall): void {
  const { agent, invocation, seq } = call;
  if (invocation === undefined) return;
  const { answered } = recorded();
  let latest = answered.get(invocation);
  if (latest === undefined) {
    latest = new Map();
    answered.set(invocation, latest);
  }
  latest.set(agent, Math.max(latest.get(agent) ?? 0, seq));
  const frame = openFrame();
  if (frame === undefined) return;
  frame.pending = frame.pending.filter(
    (recording) => recording.agent !== agent || recording.invocation !== invocation || recording.seq > seq
  );
}

/**
 * Declares that the caller handled `error` from a failed `TemporalModel` call, so it must
 * not fail the Workflow or reject the Update it happened in. Other failures absorbed by the
 * same Workflow invocation — the main function, or the Signal or Update handler that ran
 * the turn — still surface. Does nothing outside a Workflow.
 *
 * Call it from an ADK `onModelErrorCallback`, passing the `error` that callback received,
 * and return the substitute response from that callback — built with ADK's `createEvent`,
 * because ADK yields the callback's return value untouched and `isFinalResponse` then
 * dereferences the `actions` a bare `LlmResponse` does not carry. ADK finishes the run on
 * the substitute, and without this call the Workflow fails anyway and the recovery is
 * discarded. A cancellation cannot be handled this way: an execution whose cancel a model
 * call absorbed still ends CANCELLED.
 */
export function markModelFailureHandled(error: unknown): void {
  if (!inWorkflowContext()) return;
  const frame = openFrame();
  if (frame === undefined) return;
  const at = frame.pending.findIndex((recording) => recording.error === error);
  if (at !== -1) frame.pending.splice(at, 1);
}

function raiseAbsorbed(frame: Frame): void {
  // Only the execution's own cancellation is recorded (see `recordCancellation`).
  if (frame.cancellation !== undefined) {
    // Re-raised as caught: CANCELLED is read off the original `ActivityFailure` /
    // `CancelledFailure` pair, which any wrapper would hide.
    throw frame.cancellation;
  }
  if (frame.pending.length > 0) throw frame.pending[0]!.error;
}

/**
 * The name of ADK's `NodeReportedError`: a graph node that ended without output because
 * the agent it ran absorbed an error. When that error was a `TemporalModel` call the frame
 * has it recorded, and the recording — an `ActivityFailure` carrying the model failure's
 * status and cause chain — is the better thing to raise.
 */
const NODE_REPORTED_ERROR = 'NodeReportedError';

/**
 * The name of ADK's `DynamicNodeFailError`: the carrier a dynamic node (`ctx.runNode`)
 * wraps whatever its child threw in, on `.error` rather than on `.cause`. A static node
 * rethrows its child's error as it is, so the wrapper is unwrapped here to keep the two
 * alike.
 */
const DYNAMIC_NODE_FAIL_ERROR = 'DynamicNodeFailError';

/**
 * Follows ADK's dynamic-node carriers down to the error a child actually raised. Nesting
 * one dynamic run inside another wraps the carrier again, and the outer ones say nothing
 * about what went wrong. `.error` is always an `Error` where ADK sets it; the guards make
 * a hand-built or cyclic carrier terminate rather than spin.
 */
function innermostNodeFailure(err: Error): Error {
  const seen = new Set<Error>([err]);
  let current = err;
  while (current.name === DYNAMIC_NODE_FAIL_ERROR) {
    const wrapped: unknown = (current as { error?: unknown }).error;
    if (!(wrapped instanceof Error) || seen.has(wrapped)) break;
    seen.add(wrapped);
    current = wrapped;
  }
  return current;
}

/**
 * Converts an ADK workflow-runtime error escaping a frame into the failure that should
 * end the execution. A dynamic node's carrier is unwrapped first, so the decision is the
 * one a static node would have produced: a Temporal failure the child raised is returned
 * as it is, an absorbed model call is re-raised from the frame's recording, and anything
 * else becomes a non-retryable `ApplicationFailure` typed per
 * {@link ADK_RUNTIME_FAILURE_TYPES} with the ADK error as its cause. A cancellation the
 * frame recorded outranks all of that once the execution itself is cancelled. Anything the
 * map does not name — a `TemporalFailure`, a user's own error — is returned unchanged.
 */
function toWorkflowFailure(err: unknown, frame: Frame): unknown {
  if (!(err instanceof Error)) return err;
  const type = ADK_RUNTIME_FAILURE_TYPES[err.name];
  if (type === undefined) return err;
  // A model call that a cancel ended was recorded apart from the failures, and ADK then
  // absorbed it like any model error and reported the node as failed (`NodeReportedError`,
  // possibly inside a dynamic run's carrier). When the execution itself was cancelled,
  // the recorded cancellation is the outcome, by the same rule `raiseAbsorbed` applies on
  // a normal return; converting ADK's report would end a cancelled execution FAILED.
  if (frame.cancellation !== undefined) {
    return frame.cancellation;
  }
  const cause = innermostNodeFailure(err);
  // A Temporal failure the child raised is what must end the execution: a cancelled
  // Activity has to end it CANCELLED rather than FAILED, and a failed one keeps the
  // cause chain (status, retry state) that the carrier would hide.
  if (cause instanceof TemporalFailure) return cause;
  if (cause.name === NODE_REPORTED_ERROR && frame.pending.length > 0) return frame.pending[0]!.error;
  // The innermost error names the failure; the carrier's own type is the fallback for a
  // child error the map does not know.
  return ApplicationFailure.create({
    message: cause.message,
    type: ADK_RUNTIME_FAILURE_TYPES[cause.name] ?? type,
    nonRetryable: true,
    cause,
  });
}

/**
 * Hands a cancellation a Signal handler absorbed to the main function's frame when the
 * handler returns, so that the main function's own cancellation handling and cleanup run
 * before the execution ends CANCELLED. A Signal handler that raised it would end the
 * execution at once. The cancellation outranks the failures the handler also absorbed, so
 * it moves as the handler's whole outcome: none of them is left to raise in its place.
 * Kept by the handler's frame once the main function has returned.
 */
function deferToMain(frame: Frame): void {
  const { main } = recorded();
  if (frame.cancellation === undefined || main === undefined || main === frame || main.surfaced) return;
  main.cancellation ??= frame.cancellation;
  frame.cancellation = undefined;
  frame.pending = [];
}

async function surfaceAbsorbedFailure<T>(
  frame: Frame,
  next: () => Promise<T>,
  onReturn?: (frame: Frame) => void
): Promise<T> {
  let result: T;
  try {
    result = await recorded().frames.run(frame, next);
  } catch (err) {
    // `continueAsNew()` ends the run without surfacing what was absorbed, so the chain
    // would reach a successful terminal state; it ends the execution wherever it is called
    // from, so the main function's failures count too.
    if (err instanceof ContinueAsNew) {
      raiseAbsorbed(frame);
      const { main } = recorded();
      if (main !== undefined && main !== frame) raiseAbsorbed(main);
    }
    throw toWorkflowFailure(err, frame);
  } finally {
    frame.surfaced = true;
  }
  onReturn?.(frame);
  raiseAbsorbed(frame);
  return result;
}

// ts-prune-ignore-next (loaded by path from workflowInterceptorModules)
export const interceptors: WorkflowInterceptorsFactory = () => {
  watchExecutionCancel();
  return {
    inbound: [
      {
        execute: (input, next) => {
          const frame: Frame = { pending: [], surfaced: false };
          recorded().main = frame;
          return surfaceAbsorbedFailure(frame, () => next(input));
        },
        handleUpdate: (input, next) => surfaceAbsorbedFailure({ pending: [], surfaced: false }, () => next(input)),
        handleSignal: (input, next) =>
          surfaceAbsorbedFailure({ pending: [], surfaced: false }, () => next(input), deferToMain),
      },
    ],
    outbound: [
      {
        scheduleActivity: (input, next) => watched(next(input)),
        scheduleLocalActivity: (input, next) => watched(next(input)),
        startChildWorkflowExecution: (input, next) => watchedEach(next(input), (promises) => promises),
        startNexusOperation: (input, next) => watchedEach(next(input), ({ result }) => [result]),
      },
    ],
  };
};
