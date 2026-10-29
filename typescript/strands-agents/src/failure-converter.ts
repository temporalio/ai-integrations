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

/**
 * Put Strands' rewrite of interrupt and terminal errors in front of the caller's failure converter,
 * so both apply. {@link StrandsPlugin} loads this through a generated module when the client or
 * worker already has a `failureConverterPath`. Decoding uses the caller's converter.
 *
 * The caller's converter is called as it is, on itself, so its own fields and delegation keep
 * working, and the thrown errors are not changed or copied. Strands' rewrite of a nested `cause`
 * is then applied to the failure the caller returned: each `cause` failure that comes from a
 * Strands error is replaced with the caller's encoding of the rewritten error, following the
 * causes the SDK encodes.
 */
export function composeFailureConverter(caller: FailureConverter): FailureConverter {
  function withRewrittenCauses(
    err: unknown,
    failure: ProtoFailure,
    payloadConverter: PayloadConverter,
    context?: SerializationContext
  ): ProtoFailure {
    // The SDK encodes a failure decoded earlier as it was received, without its error's causes.
    if (err === null || typeof err !== 'object' || (err instanceof TemporalFailure && err.failure)) return failure;
    const cause: unknown = (err as { cause?: unknown }).cause;
    if (!cause || !failure.cause) return failure;
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
