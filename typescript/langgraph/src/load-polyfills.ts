/**
 * Global polyfills LangGraph.js needs in the Temporal Workflow sandbox.
 *
 * Installed ONLY inside Workflow context (gated on `inWorkflowContext()`), so
 * a normal Node import (the Worker and Activity side, tests) is left
 * untouched. The `./workflow` entry point imports this module for its side
 * effect, and `LangGraphPlugin.configureBundler` additionally puts it first in
 * `workflowInterceptorModules`, which guarantees per-Workflow evaluation
 * before the user's Workflow module regardless of import order.
 *
 * - `crypto`: see `crypto-shim.ts`.
 * - `ReadableStream`, `WritableStream`, `TransformStream`: `@langchain/core`
 *   builds its stream utilities on them at module load.
 * - `queueMicrotask`: the Pregel loop schedules with it.
 * - The LangChain `AsyncLocalStorage` singleton: the sandbox bundle resolves
 *   `@langchain/langgraph` to its web entry, which doesn't initialize it, and
 *   `interrupt()`, `getConfig()` and the Functional API need it.
 *
 * @module
 * @internal
 */

import { inWorkflowContext, type WorkflowInterceptorsFactory } from '@temporalio/workflow';

import { AsyncLocalStorage } from './async-hooks-shim';
import { webcrypto } from './crypto-shim';

/** The interceptor-module contract; the module is listed for its side effects. */
export const interceptors: WorkflowInterceptorsFactory = () => ({});

if (inWorkflowContext()) {
  const globals = globalThis as Record<string, unknown>;

  // Unlike the other globals here it does not yield to one the sandbox might
  // add: IDs recorded in history must keep coming from the same stream.
  globals.crypto = webcrypto;

  if (typeof globals.ReadableStream === 'undefined') {
    // eslint-disable-next-line @typescript-eslint/no-require-imports,import/no-unassigned-import
    require('web-streams-polyfill/polyfill');
  }

  if (typeof globals.queueMicrotask === 'undefined') {
    globals.queueMicrotask = (callback: () => void) => {
      void Promise.resolve().then(callback);
    };
  }

  // Required after the globals above: `@langchain/core` reads them at module load.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { AsyncLocalStorageProviderSingleton } = require('@langchain/core/singletons');
  AsyncLocalStorageProviderSingleton.initializeGlobalInstance(new AsyncLocalStorage());
}
