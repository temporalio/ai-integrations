/**
 * Deterministic `crypto` for LangGraph.js inside the Workflow sandbox.
 *
 * `@langchain/core`'s vendored `uuid` (task and checkpoint IDs) calls
 * `crypto.randomUUID` for v4 IDs and `crypto.getRandomValues` for the others,
 * and the sandbox has no `crypto` global. The polyfill loader
 * (`load-polyfills.ts`) installs {@link webcrypto} as `globalThis.crypto`.
 *
 * Every value comes from a named Workflow random stream, seeded per run, so
 * the IDs replay identically without consuming from the Workflow's own
 * `Math.random()` sequence. These values are NOT cryptographically random, and
 * the global deliberately has no `subtle`.
 *
 * `getRandomStream` asserts Workflow context, so it is called lazily.
 *
 * This module is part of the Workflow-sandbox import graph: it must not import
 * any Worker-only module.
 *
 * @module
 * @internal
 */

import { getRandomStream, type WorkflowRandomStream } from '@temporalio/workflow';

/** The package name, so a user's own `getRandomStream('…')` is unaffected by how many IDs LangGraph draws. */
const STREAM_NAME = '@temporalio/langgraph';

function stream(): WorkflowRandomStream {
  return getRandomStream(STREAM_NAME);
}

/** An RFC 4122 version-4 UUID drawn from the plugin's Workflow random stream. */
export function randomUUID(): string {
  return stream().uuid4();
}

/**
 * Fills `view` in place with bytes from the plugin's Workflow random stream and
 * returns it, as WebCrypto's `getRandomValues` does.
 */
export function getRandomValues<T extends ArrayBufferView>(view: T): T {
  stream().fill(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
  return view;
}

/** The object the polyfill loader installs as the sandbox's `crypto` global. */
export const webcrypto = { randomUUID, getRandomValues };
