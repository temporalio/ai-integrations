/**
 * Web/Node global polyfills for the Temporal Workflow sandbox.
 *
 * `@google/adk` and `@google/genai` reference `Headers`, `structuredClone`,
 * and the WHATWG streams globals while building requests in the agent loop,
 * and ADK's telemetry modules reach `@opentelemetry/core`'s browser build,
 * which dereferences the `performance` global at module load. The Workflow
 * sandbox does not expose all of them, so we install minimal polyfills — but
 * ONLY inside Workflow context (gated on `inWorkflowContext()`), so a normal
 * Node import (the worker / Activity side, tests, direct ADK use) is left
 * untouched. The `./workflow` entry point imports this module for its side
 * effect, and `GoogleAdkPlugin.configureBundler` additionally prepends it to
 * `workflowInterceptorModules`, which guarantees per-workflow evaluation
 * before the user's workflow module regardless of user import order.
 */

import { inWorkflowContext, type WorkflowInterceptorsFactory } from '@temporalio/workflow';

import { webcrypto } from './crypto-shim';

// Satisfies the documented interceptor-module contract (modules on
// `workflowInterceptorModules` export an `interceptors` factory); this module
// is on that list purely for its load-time side effect, so it registers none.
export const interceptors: WorkflowInterceptorsFactory = () => ({});

if (inWorkflowContext()) {
  const globals = globalThis as Record<string, unknown>;

  if (typeof globals.Headers === 'undefined') {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { Headers } = require('headers-polyfill');
    globals.Headers = Headers;
  }

  if (typeof globals.structuredClone === 'undefined') {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    globals.structuredClone = require('@ungap/structured-clone').default;
  }

  // ADK's id generator (`utils/env_aware_utils.js`, `randomUUID()`) reads
  // `globalThis.crypto.randomUUID` / `.getRandomValues` before falling back
  // to its `node:crypto` import, the `uuid` package ADK's retry plugins
  // import reads only the global, and ADK 2.1+'s single-file web bundle has
  // `node:crypto` pre-resolved to a stand-in that throws. So the plugin's
  // deterministic shim is also the sandbox's `crypto` global: `randomUUID`
  // and `getRandomValues` only (no `subtle`), every value drawn from the
  // plugin's named workflow random stream. It is the same draw the bundler
  // plugin's `node:crypto` redirect serves, so ids regenerate identically
  // under replay whichever path ADK takes. Unlike the other globals here it
  // does not yield to one the sandbox already has (it has none today): the
  // ids ADK draws, which a human-in-the-loop answer is matched against under
  // replay, must keep coming from this stream whatever a later sandbox or
  // SDK puts there.
  globals.crypto = webcrypto;

  if (typeof globals.ReadableStream === 'undefined') {
    // eslint-disable-next-line @typescript-eslint/no-require-imports,import/no-unassigned-import
    require('web-streams-polyfill/polyfill');
  }

  if (typeof globals.performance === 'undefined') {
    // `@opentelemetry/core`'s browser build (reached from ADK's telemetry
    // modules through `@opentelemetry/sdk-trace-base`) evaluates
    // `export const otperformance = performance` at module load, and span
    // timestamps flow through `performance.timeOrigin` / `performance.now()`.
    // Both map onto the sandbox-patched `Date.now()` (workflow time), so the
    // values are deterministic under replay. Mirrors the shim
    // `@temporalio/interceptors-opentelemetry` installs from its own workflow
    // runtime module; ours yields to an existing shim, while that package's
    // unconditional `Object.assign` may later replace ours — harmless either
    // way, since both compute identical values from the same sandbox clock.
    const timeOrigin = Date.now();
    globals.performance = {
      timeOrigin,
      now() {
        return Date.now() - timeOrigin;
      },
    };
  }
}
