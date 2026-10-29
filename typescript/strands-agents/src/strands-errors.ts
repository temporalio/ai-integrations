import { ContextWindowOverflowError, MaxTokensError, StructuredOutputError } from '@strands-agents/sdk';
import { ApplicationFailure, ApplicationFailureCategory } from '@temporalio/common';

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

/**
 * Rewrite a Strands terminal model error as a typed, non-retryable
 * {@link ApplicationFailure} that keeps its message, stack and cause. Any other
 * value is returned as it is.
 */
export function rewriteTerminalError(err: unknown): unknown {
  if (err instanceof Error) {
    for (const { ctor, name } of TERMINAL_ERROR_TYPES) {
      // Match the name too, in case another copy of `@strands-agents/sdk` threw it.
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
 * benign failure, so the worker does not log it as an activity failure. Any
 * other value is returned as it is.
 */
export function rewriteError(err: unknown): unknown {
  if (isInterruptError(err)) {
    const details = err.interrupts.map((i) => i.toJSON());
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
