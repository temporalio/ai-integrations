/**
 * Bundle-time stand-in for `node:async_hooks` inside the Workflow isolate,
 * where that builtin is unavailable. The plugin's bundler replaces
 * `async_hooks` and `node:async_hooks` with this module (see `sandboxCompat`
 * in `plugin.ts`), so `@langchain/core`'s and `@langchain/langgraph`'s
 * `AsyncLocalStorage` imports resolve to the async-context-tracking
 * `AsyncLocalStorage` the SDK injects onto the Workflow-sandbox global.
 *
 * @module
 * @internal
 */

declare const globalThis: { AsyncLocalStorage: typeof import('node:async_hooks').AsyncLocalStorage };

export const AsyncLocalStorage = globalThis.AsyncLocalStorage;
