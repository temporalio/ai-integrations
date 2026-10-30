import { ContextWindowOverflowError, MaxTokensError, StructuredOutputError } from '@strands-agents/sdk';
import { ApplicationFailure, ApplicationFailureCategory } from '@temporalio/common';

export const STRANDS_INTERRUPT_TYPE = 'StrandsInterrupt';

// Strands does not re-export `InterruptError` from its public index, so detect
// it structurally instead of via `instanceof`.
interface InterruptLike {
  toJSON(): unknown;
}

interface InterruptDetails {
  id?: string;
  name: string;
  reason?: unknown;
  response?: unknown;
  source?: string;
}

function isInterruptError(err: unknown): err is { interrupts: unknown[] } {
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

/**
 * Rewrite a Strands terminal model error as a typed, non-retryable
 * {@link ApplicationFailure} that keeps its message, stack and cause. Any other
 * value is returned as it is.
 */
export function rewriteTerminalError(err: unknown): unknown {
  if (err instanceof Error) {
    for (const { ctor, name } of TERMINAL_ERROR_TYPES) {
      // The name matches an error from another copy of `@strands-agents/sdk`, or any error with
      // that name; `instanceof` also matches a subclass that sets a name of its own.
      if (err instanceof ctor || err.name === name) {
        return ApplicationFailure.fromError(err, { type: name, nonRetryable: true });
      }
    }
  }
  return err;
}

/**
 * Rewrite a Strands interrupt or terminal model error as a typed, non-retryable
 * {@link ApplicationFailure} that keeps its stack and cause. An interrupt is a
 * benign failure, so the worker does not log it as an activity failure; a
 * malformed one becomes a non-retryable failure that says so. Any other value is
 * returned as it is.
 */
export function rewriteError(err: unknown): unknown {
  if (isInterruptError(err)) {
    const details = interruptDetails(err.interrupts);
    if (details === undefined) {
      // Activities build interrupts by hand. Fail a malformed one without retries, rather than
      // retrying it with no pause or stopping the worker when its details cannot be encoded.
      return ApplicationFailure.fromError(err, {
        message:
          'malformed interrupt: each entry needs a toJSON() that returns an object with a string name, and JSON-encodable values',
        nonRetryable: true,
      });
    }
    return ApplicationFailure.fromError(err, {
      message: `interrupt:${details.map((d) => d.name).join(',')}`,
      type: STRANDS_INTERRUPT_TYPE,
      nonRetryable: true,
      details,
      category: ApplicationFailureCategory.BENIGN,
    });
  }
  return rewriteTerminalError(err);
}

// The details of each interrupt, or undefined when an entry has no toJSON(), returns no object
// with a string name, or holds a value the default payload converter cannot encode as JSON.
function interruptDetails(interrupts: unknown[]): InterruptDetails[] | undefined {
  try {
    const details = interrupts.map((i) => (i as InterruptLike).toJSON());
    const named = details.every(
      (d) => d !== null && typeof d === 'object' && typeof (d as { name?: unknown }).name === 'string'
    );
    if (!named) return undefined;
    JSON.stringify(details);
    return details as InterruptDetails[];
  } catch {
    return undefined;
  }
}
