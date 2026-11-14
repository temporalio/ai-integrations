import { Headers } from 'headers-polyfill';
import { DeterminismViolationError, inWorkflowContext, uuid4 } from '@temporalio/workflow';

if (inWorkflowContext()) {
  // Strands composes the parent's cancellation signal with the subagent's via AbortSignal.any().
  // Keep native signal composition, but refuse timeout's timer outside Temporal history.
  // Subclass instead of changing the constructor shared with the worker and other workflows.
  if (typeof globalThis.AbortSignal === 'undefined') {
    const NativeAbortSignal = new AbortController().signal.constructor as typeof AbortSignal;
    globalThis.AbortSignal = class extends NativeAbortSignal {
      static override timeout(_milliseconds: number): never {
        throw new DeterminismViolationError(
          'AbortSignal.timeout() is not supported in workflows. Use Temporal sleep() and AbortController instead.'
        );
      }

      // AbortController and the inherited helpers still return native signals.
      static override [Symbol.hasInstance](value: unknown): boolean {
        return value instanceof NativeAbortSignal;
      }
    };
  }

  if (typeof globalThis.Headers === 'undefined') {
    globalThis.Headers = Headers;
  }

  // eslint-disable-next-line @typescript-eslint/no-require-imports,import/no-unassigned-import
  require('web-streams-polyfill/polyfill');

  if (!('structuredClone' in globalThis)) {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const structuredClone = require('@ungap/structured-clone');
    globalThis.structuredClone = structuredClone.default;
  }

  // `@ungap/structured-clone` calls `crypto.randomUUID()` internally, but the
  // workflow sandbox bans the real `crypto` module. Use the SDK's
  // deterministic `uuid4` so the polyfill still works at replay.
  if (typeof globalThis.crypto?.randomUUID !== 'function') {
    globalThis.crypto = { ...(globalThis.crypto ?? {}), randomUUID: uuid4 } as never;
  }
}
