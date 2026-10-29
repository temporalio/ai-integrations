import { ContextWindowOverflowError, MaxTokensError, StructuredOutputError } from '@strands-agents/sdk';
import {
  ApplicationFailure,
  DefaultFailureConverter,
  type FailureConverter,
  type PayloadConverter,
  type ProtoFailure,
  type SerializationContext,
  TemporalFailure,
} from '@temporalio/common';

export const STRANDS_INTERRUPT_TYPE = 'StrandsInterrupt';

// Strands does not re-export `InterruptError` from its public index, so detect
// it structurally instead of via `instanceof`.
interface InterruptLike {
  toJSON(): { id: string; name: string; reason?: unknown; response?: unknown; source: string };
}

function isInterruptError(err: unknown): err is { interrupts: InterruptLike[] } {
  if (err === null || typeof err !== 'object') return false;
  const candidate = err as { interrupts?: unknown };
  if (!Array.isArray(candidate.interrupts) || candidate.interrupts.length === 0) return false;
  const first = candidate.interrupts[0] as { toJSON?: unknown } | undefined;
  return typeof first?.toJSON === 'function';
}

const TERMINAL_ERROR_TYPES = [
  { ctor: ContextWindowOverflowError, name: 'ContextWindowOverflowError' },
  { ctor: MaxTokensError, name: 'MaxTokensError' },
  { ctor: StructuredOutputError, name: 'StructuredOutputError' },
];

function rewriteError(err: unknown): unknown {
  if (isInterruptError(err)) {
    const details = err.interrupts.map((i) => i.toJSON());
    const names = err.interrupts.map((i) => i.toJSON().name).join(',');
    return ApplicationFailure.create({
      message: `interrupt:${names}`,
      type: STRANDS_INTERRUPT_TYPE,
      nonRetryable: true,
      details,
    });
  }
  if (err instanceof Error) {
    for (const { ctor, name } of TERMINAL_ERROR_TYPES) {
      if (err instanceof ctor) {
        return ApplicationFailure.create({
          message: err.message,
          type: name,
          nonRetryable: true,
        });
      }
    }
  }
  return err;
}

/**
 * Failure converter that preserves Strands `InterruptError` payloads across the
 * activity boundary as typed, non-retryable {@link ApplicationFailure}s and
 * marks deterministic Strands model errors as non-retryable.
 *
 * Read by the worker and client via {@link DataConverter.failureConverterPath};
 * the named export below is what {@link StrandsPlugin} wires in.
 */
export class StrandsFailureConverter extends DefaultFailureConverter implements FailureConverter {
  override errorToFailure(
    err: unknown,
    payloadConverter: PayloadConverter,
    context?: SerializationContext
  ): ProtoFailure {
    return super.errorToFailure(rewriteError(err), payloadConverter, context);
  }
}

// ts-prune-ignore-next
export const failureConverter: FailureConverter = new StrandsFailureConverter();

// Whether the SDK encodes the cause of `err`. It does not for non-objects, and it encodes a failure
// decoded earlier as it was received, without its error's causes.
function encodesCause(err: unknown): err is { cause?: unknown } {
  return err !== null && typeof err === 'object' && !(err instanceof TemporalFailure && err.failure);
}

// Whether a Strands error is in the cause chain of `err` that `failure` encodes, as far as both go.
function hasStrandsCause(err: unknown, failure: ProtoFailure): err is { cause: unknown } {
  let e = err;
  let f = failure;
  while (encodesCause(e) && e.cause && f.cause) {
    if (rewriteError(e.cause) !== e.cause) return true;
    e = e.cause;
    f = f.cause;
  }
  return false;
}

// Whether two failures are the same: equal primitives and bytes, and objects of the same prototype
// with the same own enumerable properties.
function sameFailure(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a === null || b === null || typeof a !== 'object' || typeof b !== 'object') return false;
  if (Object.getPrototypeOf(a) !== Object.getPrototypeOf(b)) return false;
  if (a instanceof Uint8Array) {
    return b instanceof Uint8Array && a.length === b.length && a.every((byte, i) => byte === b[i]);
  }
  const aRecord = a as Record<string, unknown>;
  const bRecord = b as Record<string, unknown>;
  const keys = Object.keys(aRecord);
  return (
    keys.length === Object.keys(bRecord).length &&
    keys.every((key) => Object.prototype.hasOwnProperty.call(bRecord, key) && sameFailure(aRecord[key], bRecord[key]))
  );
}

/**
 * Put Strands' rewrite of interrupt and terminal errors in front of the caller's failure converter,
 * so both apply. {@link StrandsPlugin} loads this through a generated module when the client or
 * worker already has a `failureConverterPath`. Decoding uses the caller's converter.
 *
 * The caller's converter is called as it is, on itself, so its own fields and delegation keep
 * working, and the thrown errors are not changed or copied. Strands' rewrite of a nested `cause`
 * is then applied to the failure the caller returned: each `cause` failure that comes from a
 * Strands error is replaced with the caller's encoding of the rewritten error, following the
 * causes the SDK encodes. A `cause` failure is only replaced or followed when it is the same as the
 * caller's own encoding of the error's `cause`, so failures the caller adds, drops or changes
 * between levels are kept as the caller returned them, without the nested rewrite.
 */
export function composeFailureConverter(caller: FailureConverter): FailureConverter {
  function withRewrittenCauses(
    err: unknown,
    failure: ProtoFailure,
    payloadConverter: PayloadConverter,
    context?: SerializationContext
  ): ProtoFailure {
    if (!hasStrandsCause(err, failure) || !failure.cause) return failure;
    const cause = err.cause;
    if (!sameFailure(failure.cause, caller.errorToFailure(cause, payloadConverter, context))) return failure;
    const rewritten = rewriteError(cause);
    const causeFailure =
      rewritten === cause
        ? withRewrittenCauses(cause, failure.cause, payloadConverter, context)
        : caller.errorToFailure(rewritten, payloadConverter, context);
    // Copy rather than assign: the caller may return a failure it keeps, such as a decoded one.
    return causeFailure === failure.cause ? failure : { ...failure, cause: causeFailure };
  }
  return {
    errorToFailure(err, payloadConverter, context) {
      const rewritten = rewriteError(err);
      const failure = caller.errorToFailure(rewritten, payloadConverter, context);
      return withRewrittenCauses(rewritten, failure, payloadConverter, context);
    },
    failureToError(failure, payloadConverter, context) {
      return caller.failureToError(failure, payloadConverter, context);
    },
  };
}
