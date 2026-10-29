import { runInNewContext } from 'node:vm';
import type { TestFn } from 'ava';
import { StructuredOutputError } from '@strands-agents/sdk';
import { Client, WorkflowFailedError } from '@temporalio/client';
import {
  ApplicationFailure,
  DefaultFailureConverter,
  defaultPayloadConverter,
  type FailureConverter,
  type PayloadConverter,
  type ProtoFailure,
  type RetryPolicy,
  type SerializationContext,
} from '@temporalio/common';
import type { BaseContext } from '@temporalio/test-helpers';
import { test as anyTest, helpers, TestWorkflowEnvironment } from '@temporalio/test-helpers';
import { StrandsFailureConverter, StrandsPlugin } from '..';
import { composeFailureConverter } from '../failure-converter';
import { failureConverter as envelopingConverter } from './encoding-failure-converter';
import { startLocalWithRetry } from './helpers/environment';
import { failFrozen, failTwice, lastFailureOfRetry } from './workflows/strands';

// Its own file, so the second workflow bundle this test builds lives in its own test process and
// does not add to the memory the main suite's concurrent workers use.
const test = anyTest as TestFn<BaseContext>;

test.before(async (t) => {
  t.context = { env: await startLocalWithRetry(() => TestWorkflowEnvironment.createLocal()) } as BaseContext;
});

test.after.always(async (t) => {
  await t.context.env?.teardown();
});

// Activity failure causes that Strands must rewrite: a StructuredOutputError, and errors the SDK
// encodes as errors although they are not `Error`s of this realm, each with a StructuredOutputError
// cause. A DOMException's accessors only work on the DOMException itself.
const activityCauses: Record<string, () => unknown> = {
  direct: () => new StructuredOutputError('no output'),
  domException: () =>
    Object.defineProperty(new DOMException('wrapper', 'DataError'), 'cause', {
      value: new StructuredOutputError('no output'),
    }),
  structural: () => ({
    name: 'Error',
    message: 'wrapper',
    stack: 'Error: wrapper',
    cause: new StructuredOutputError('no output'),
  }),
  otherRealm: () =>
    Object.assign(runInNewContext('new Error("wrapper")') as Error, { cause: new StructuredOutputError('no output') }),
};

// A failure the caller's converter in `encoding-failure-converter` wraps in a failure of its own, with
// a Strands cause.
const businessError = (): ApplicationFailure =>
  ApplicationFailure.create({
    message: 'business outer',
    type: 'Business',
    nonRetryable: true,
    details: ['business detail'],
    cause: new StructuredOutputError('no output'),
  });

test("a worker's own failure converter still applies with the plugin", async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const failureConverterPath = require.resolve('./encoding-failure-converter');

  // The worker bundles the workflows itself, so the bundle gets its failure converter.
  const worker = await createWorker({
    workflowsPath: require.resolve('./workflows/strands'),
    dataConverter: { failureConverterPath },
    plugins: [new StrandsPlugin()],
    activities: {
      async failingActivity(message: string): Promise<never> {
        if (message === 'business') throw businessError();
        throw ApplicationFailure.create({
          message: `activity failed: ${message}`,
          nonRetryable: true,
          cause: activityCauses[message]?.() as Error | undefined,
        });
      },
    },
  });
  const client = new Client({
    connection: t.context.env.connection,
    namespace: t.context.env.namespace,
    dataConverter: { failureConverterPath },
    plugins: [new StrandsPlugin()],
  });
  const [history, frozenHistory, lastFailure] = await worker.runUntil(async () => {
    const handle = await startWorkflow(failTwice, {
      args: ['workflow failed', ['plain', ...Object.keys(activityCauses), 'business']],
    });
    // A client with the same converter decodes the failure and its rewritten cause.
    const err = await t.throwsAsync(client.workflow.getHandle(handle.workflowId).result(), {
      instanceOf: WorkflowFailedError,
    });
    const decodedCause = err?.cause?.cause;
    t.deepEqual(
      [
        err?.cause?.message,
        decodedCause instanceof ApplicationFailure && [
          decodedCause.type,
          decodedCause.nonRetryable,
          decodedCause.message,
        ],
      ],
      ['workflow failed', ['StructuredOutputError', true, 'no output']]
    );
    // Workflow code decoded the caller's wrapped failure as the caller's converter alone does.
    const decodedInWorkflow = (err?.cause as ApplicationFailure | undefined)?.details?.[0] as
      Record<string, unknown> | undefined;
    t.deepEqual(decodedInWorkflow?.business, [
      ['Envelope', 'envelope', []],
      ['Business', 'business outer', ['business detail']],
      'no output',
    ]);
    // A frozen failure's `cause` cannot be assigned; Strands' rewrite of it must not fail the
    // workflow task, which would leave the run open until it times out.
    const frozen = await startWorkflow(failFrozen, {
      args: ['frozen failure'],
      workflowExecutionTimeout: '30 seconds',
    });
    await t.throwsAsync(frozen.result());
    // The SDK decodes a retried run's last failure before the run starts; it still gets the
    // caller's converter.
    const retry: RetryPolicy = { maximumAttempts: 2, initialInterval: '10 milliseconds' };
    const retried = await startWorkflow(lastFailureOfRetry, { retry });
    return [await handle.fetchHistory(), await frozen.fetchHistory(), await retried.result()] as const;
  });

  // All failures were stored as the caller's converter writes them, and Strands still marked the
  // workflow failure's StructuredOutputError cause as non-retryable.
  const workflowFailure = history.events?.find((e) => e.workflowExecutionFailedEventAttributes)
    ?.workflowExecutionFailedEventAttributes?.failure;
  const cause = workflowFailure?.cause?.applicationFailureInfo;
  t.deepEqual(
    [workflowFailure?.message, cause?.type, cause?.nonRetryable, lastFailure],
    ['Encoded failure', 'StructuredOutputError', true, 'decoded by the caller']
  );
  // Each activity failure was recorded, and Strands rewrote each StructuredOutputError cause, also
  // inside each wrapper.
  const activityFailures = (history.events ?? []).flatMap((e) => {
    const failure = e.activityTaskFailedEventAttributes?.failure;
    if (!failure) return [];
    const chain = [failure];
    for (let f = failure.cause; f; f = f.cause) chain.push(f);
    const inner = chain.length > 1 ? chain[chain.length - 1]?.applicationFailureInfo : undefined;
    return [[chain.length, chain.every((f) => f.message === 'Encoded failure'), inner?.type, inner?.nonRetryable]];
  });
  const wrapped = [3, true, 'StructuredOutputError', true];
  t.deepEqual(activityFailures, [
    [1, true, undefined, undefined],
    [2, true, 'StructuredOutputError', true],
    wrapped,
    wrapped,
    wrapped,
    [3, true, undefined, undefined],
  ]);
  // The caller wrapped that last failure in an envelope; Strands kept the envelope and the failure it
  // wraps, with its type and details, and did not rewrite its cause.
  const business = history.events
    ?.map((e) => e.activityTaskFailedEventAttributes?.failure)
    .find((f) => f?.applicationFailureInfo?.type === 'Envelope');
  t.deepEqual(
    [
      business?.cause?.applicationFailureInfo?.type,
      defaultPayloadConverter.fromPayload(business?.cause?.applicationFailureInfo?.details?.payloads?.[0] ?? {}),
      business?.cause?.cause?.applicationFailureInfo ?? undefined,
    ],
    ['Business', 'business detail', undefined]
  );
  // The frozen failure was stored the same way, with its cause rewritten.
  const frozenFailure = frozenHistory.events?.find((e) => e.workflowExecutionFailedEventAttributes)
    ?.workflowExecutionFailedEventAttributes?.failure;
  const frozenCause = frozenFailure?.cause?.applicationFailureInfo;
  t.deepEqual(
    [frozenFailure?.message, frozenCause?.type, frozenCause?.nonRetryable],
    ['Encoded failure', 'StructuredOutputError', true]
  );
});

// Caller converters of other shapes, each encoding as the SDK default does.
class PrivateFieldFailureConverter extends DefaultFailureConverter {
  readonly #prefix = '';

  override errorToFailure(err: unknown, payloadConverter: PayloadConverter, context?: SerializationContext) {
    const failure = super.errorToFailure(err, payloadConverter, context);
    return { ...failure, message: this.#prefix + (failure.message ?? '') };
  }
}

function delegatingFrozen(inner: FailureConverter): FailureConverter {
  return Object.freeze({
    errorToFailure: (err: unknown, payloadConverter: PayloadConverter, context?: SerializationContext) =>
      inner.errorToFailure(err, payloadConverter, context),
    failureToError: (failure: ProtoFailure, payloadConverter: PayloadConverter, context?: SerializationContext) =>
      inner.failureToError(failure, payloadConverter, context),
  });
}

test("Strands' rewrite applies to every cause, whatever the caller's converter looks like", (t) => {
  const strands = new StrandsFailureConverter();
  const decoded = strands.failureToError(
    strands.errorToFailure(
      ApplicationFailure.create({ message: 'decoded', cause: new StructuredOutputError('no output') }),
      defaultPayloadConverter
    ),
    defaultPayloadConverter
  );
  const errors: Record<string, () => unknown> = {
    top: () => new StructuredOutputError('no output'),
    ordinary: () => ApplicationFailure.create({ message: 'outer', cause: new StructuredOutputError('no output') }),
    frozen: () =>
      Object.freeze(ApplicationFailure.create({ message: 'outer', cause: new StructuredOutputError('no output') })),
    decoded: () => decoded,
    business: businessError,
    ...Object.fromEntries(
      Object.entries(activityCauses).map(([name, cause]) => [
        name,
        () => ApplicationFailure.create({ message: 'outer', cause: cause() as Error }),
      ])
    ),
  };
  const callers: Record<string, FailureConverter> = {
    default: new DefaultFailureConverter(),
    privateField: new PrivateFieldFailureConverter(),
    delegatingFrozen: delegatingFrozen(new PrivateFieldFailureConverter()),
  };
  const summary = (failure: ProtoFailure | null | undefined): unknown[] =>
    failure
      ? [
          [failure.message, failure.applicationFailureInfo?.type, failure.applicationFailureInfo?.nonRetryable],
          ...summary(failure.cause),
        ]
      : [];
  for (const [callerName, caller] of Object.entries(callers)) {
    const composed = composeFailureConverter(caller);
    for (const [errorName, error] of Object.entries(errors)) {
      t.deepEqual(
        summary(composed.errorToFailure(error(), defaultPayloadConverter)),
        summary(strands.errorToFailure(error(), defaultPayloadConverter)),
        `${callerName}: ${errorName}`
      );
    }
  }
});

test("Strands keeps failures a caller's converter adds around a cause", (t) => {
  const composed = composeFailureConverter(envelopingConverter);
  for (const error of [businessError, () => ApplicationFailure.create({ message: 'outer', cause: businessError() })]) {
    const callerFailure = envelopingConverter.errorToFailure(error(), defaultPayloadConverter);
    const composedFailure = composed.errorToFailure(error(), defaultPayloadConverter);
    // The same failure as the caller's alone, apart from stack traces, which differ per error.
    const levels = (failure: ProtoFailure | null | undefined): unknown[] =>
      failure ? [{ ...failure, cause: undefined, encodedAttributes: undefined }, ...levels(failure.cause)] : [];
    t.deepEqual(levels(composedFailure), levels(callerFailure));
    const decoded = composed.failureToError(composedFailure, defaultPayloadConverter);
    const business = decoded instanceof ApplicationFailure && decoded.type === 'Business' ? decoded : decoded.cause;
    t.deepEqual(business instanceof ApplicationFailure && [business.type, business.message, business.details], [
      'Business',
      'business outer',
      ['business detail'],
    ]);
  }
});
