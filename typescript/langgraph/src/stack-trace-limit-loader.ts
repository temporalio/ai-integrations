/**
 * A webpack loader for `@langchain/langgraph` in the Workflow bundle.
 *
 * LangGraph sets `Error.stackTraceLimit` while constructing some errors (for
 * example `EmptyChannelError`). A Worker that reuses its V8 context freezes
 * the sandbox's `Error`, so that assignment throws in strict mode. Stack
 * depth is irrelevant to Workflow behavior, so the loader drops the
 * assignments.
 *
 * @module
 * @internal
 */

const ASSIGNMENT = /\bError\.stackTraceLimit\s*=\s*[^;,)]+/g;

export default function stackTraceLimitLoader(source: string): string {
  return source.replace(ASSIGNMENT, 'void 0');
}
