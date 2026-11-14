import { GuardrailExecutionError, ToolCallError } from '@openai/agents-core';
import { TemporalFailure } from '@temporalio/common';

/**
 * Walks nested `cause` chains and AggregateError `errors` arrays to find the
 * first TemporalFailure, and with `throughToolErrors` also the `error` of
 * upstream's ToolCallError and GuardrailExecutionError. Cycle-safe via visited set.
 */
export function unwrapTemporalFailure(error: unknown, throughToolErrors = false): TemporalFailure | undefined {
  const visited = new Set<unknown>();
  const stack: unknown[] = [error];
  while (stack.length > 0) {
    const current = stack.pop();
    if (!current || typeof current !== 'object' || visited.has(current)) continue;
    visited.add(current);
    if (current instanceof TemporalFailure) return current;
    if (current instanceof AggregateError) {
      for (const inner of current.errors) {
        stack.push(inner);
      }
    }
    if (throughToolErrors && (current instanceof ToolCallError || current instanceof GuardrailExecutionError)) {
      stack.push(current.error);
    }
    stack.push((current as any).cause);
  }
  return undefined;
}
