/**
 * Deterministic `crypto` for `@google/adk` inside the Workflow sandbox.
 *
 * ADK's id generator (`utils/env_aware_utils.ts`, `randomUUID()`) tries
 * `globalThis.crypto.randomUUID`, then `globalThis.crypto.getRandomValues`, and
 * finally the `randomUUID` it statically imports from `node:crypto`. The sandbox
 * exposes no `crypto` global and the Worker bundler aliases the `crypto` builtin
 * to an empty module, so from ADK 1.6 onward — which dropped the `Math.random()`
 * fallback ADK 1.5 had — the first `createEvent()` / `createSession()` in a
 * Workflow would throw. This module reaches ADK two ways:
 *
 *  - **As the `node:crypto` module.** The Workflow bundle pins ADK's per-file
 *    `dist/esm/common.js` surface, where that import is a real module request;
 *    the plugin's bundler plugin redirects it, when issued from inside
 *    `@google/adk` or the `uuid` package (ADK's request for it is pinned to
 *    the plugin's own copy), to this module (`nodeRandomUUID` is
 *    {@link randomUUID}; `uuid`'s node builds take {@link randomFillSync}).
 *    ADK's OAuth2 PKCE helpers import `createHash` and `randomBytes` from the
 *    same module; those are throwing stand-ins here (see below).
 *  - **As the sandbox's `crypto` global.** The polyfill loader
 *    (`load-polyfills.ts`) installs {@link webcrypto} as `globalThis.crypto`:
 *    ADK's `randomUUID()` reads that first, the `uuid` package ADK's retry
 *    plugins import has nothing else, and ADK 2.1+'s single-file web bundle
 *    (`dist/web/index_web.js`, which a consumer's own alias could still
 *    select) has `node:crypto` pre-resolved to ADK's browser stand-in, whose
 *    `randomUUID` throws `no cryptographically secure source of randomness is
 *    available` — nothing left to redirect there.
 *
 * Either path lands on the same {@link randomUUID}, so the ids are identical
 * whichever one ADK takes, and histories replay across the two.
 *
 * Every value comes from a **named** workflow random stream
 * (`getRandomStream`), which is seeded per run and replays identically without
 * consuming from the workflow's own `Math.random()` sequence. That is what
 * makes ADK's event, invocation, session and function-call ids, and
 * `RequestInput` interrupt ids, stable across replay — a human-in-the-loop
 * response is matched by id, so the id must regenerate byte-for-byte.
 *
 * These values are NOT cryptographically random. Inside a Workflow that is the
 * point (determinism); it also means ADK's OAuth2 `state` parameter and session
 * identifiers are predictable there, which is one reason credential flows are
 * unsupported in Workflows (see the README). The global deliberately has no
 * `subtle`: ADK features that need real cryptography (PKCE, the
 * `ToolCallIntegrityPlugin` HMAC) keep failing with their own clear errors.
 *
 * `getRandomStream` asserts workflow context, so it is called lazily — this
 * module is evaluated at bundle load, before any activator exists.
 *
 * IMPORTANT: this module is part of the Workflow-sandbox import graph. It must
 * not import any worker-only module.
 */

import { getRandomStream, type WorkflowRandomStream } from '@temporalio/workflow';

/**
 * The stream name is the package name: private to this plugin, so a user's
 * own `getRandomStream('…')` and the workflow's default `Math.random()` are
 * unaffected by how many ids ADK draws.
 */
const STREAM_NAME = '@temporalio/google-adk';

function stream(): WorkflowRandomStream {
  return getRandomStream(STREAM_NAME);
}

/** An RFC 4122 version-4 UUID drawn from the plugin's workflow random stream. */
export function randomUUID(): string {
  return stream().uuid4();
}

/**
 * Fills `view` in place with bytes from the plugin's workflow random stream and
 * returns it — the WebCrypto `getRandomValues` contract, for any integer typed
 * array (the fill is byte-wise over the view's underlying storage).
 */
export function getRandomValues<T extends ArrayBufferView>(view: T): T {
  stream().fill(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
  return view;
}

/**
 * `node:crypto`'s `randomFillSync`, which the `uuid` package's node build
 * imports for its v4 byte pool: fills `size` elements of `buffer` from
 * `offset` (both in elements, as in Node) with bytes from the plugin's
 * workflow random stream and returns `buffer` — the same draw its browser
 * build makes through {@link getRandomValues}.
 */
export function randomFillSync<T extends ArrayBufferView>(buffer: T, offset = 0, size?: number): T {
  const elementSize = (buffer as { BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT ?? 1;
  const byteOffset = offset * elementSize;
  const byteLength = size === undefined ? buffer.byteLength - byteOffset : size * elementSize;
  stream().fill(new Uint8Array(buffer.buffer, buffer.byteOffset + byteOffset, byteLength));
  return buffer;
}

/**
 * The Web Crypto surface the plugin provides: `node:crypto`'s `webcrypto`
 * namespace for consumers that reach it, and the object the polyfill loader
 * installs as the sandbox's `crypto` global.
 */
export const webcrypto = { randomUUID, getRandomValues };

const PKCE_UNSUPPORTED =
  "@google/adk's OAuth2 PKCE helpers cannot run inside a Temporal Workflow: a code verifier needs " +
  'cryptographically secure randomness and a code challenge a real SHA-256, and the sandbox has ' +
  'neither. Credential flows are unsupported in Workflows; acquire credentials on the Worker ' +
  '(in an Activity or an MCP toolset factory).';

/**
 * Stand-in for `node:crypto`'s `createHash`, which ADK >= 2.1's
 * `auth/oauth2/oauth2_utils.js` imports for the PKCE code challenge. webpack
 * verifies named imports at build time, so the export must exist for the
 * bundle to build; it throws when reached, mirroring ADK's own browser
 * stand-in, because a predictable or missing challenge must not be emitted.
 */
export function createHash(_algorithm: string): never {
  throw new Error(PKCE_UNSUPPORTED);
}

/**
 * Stand-in for `node:crypto`'s `randomBytes`, imported by the same PKCE
 * helpers for the code verifier. Deliberately not served from the workflow
 * random stream: that value exists to be unguessable.
 */
export function randomBytes(_size: number): never {
  throw new Error(PKCE_UNSUPPORTED);
}

export default { randomUUID, getRandomValues, randomFillSync, webcrypto, createHash, randomBytes };
