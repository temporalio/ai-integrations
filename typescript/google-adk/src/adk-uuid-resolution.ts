/**
 * Answers `@google/adk`'s undeclared `uuid` import on the Worker.
 *
 * ADK >= 2.1's `plugins/reflect_retry_model_plugin.js` (on its `common`
 * surface, so in every build) imports `uuid` without declaring it, through
 * ADK 2.3.0 at least: its single-file web bundle inlined the package, and the
 * node builds find it only where the host layout hoists a copy. Under a
 * hoisting layout (npm, pnpm's default) that is whichever copy happens to be
 * at the top, under a strict one (pnpm `hoist=false`) nothing is reachable
 * from ADK's own directory, and since the plugin imports ADK as soon as it is
 * loaded (`activities.ts`), `require('@temporalio/google-adk')` fails with
 * `Cannot find module 'uuid'` before a Worker is ever created. The plugin's
 * own `uuid` dependency, which the Workflow bundle pins ADK's request to
 * (`plugin.ts`), is no help there: a dependency of the plugin is not reachable
 * from ADK's directory either.
 *
 * So the plugin answers the request itself. Node resolves every CommonJS
 * request through `Module._resolveFilename`; this module wraps it the way the
 * resolution tools do (`tsx`, `ts-node`, Yarn PnP, `module-alias`), so that a
 * `uuid` request Node could not resolve, made from a file inside
 * `node_modules/@google/adk/`, resolves to the plugin's own copy instead.
 * Nothing else changes: a request Node resolves on its own is untouched, every
 * other failure is rethrown as is, and the answer is the copy the Workflow
 * bundle gives ADK, so both sides draw their ids from the same package.
 *
 * It is installed at load, from every entry that reaches ADK (`index`,
 * `workflow`, `testing`) ahead of that import, once per process (a second
 * copy of the plugin finds the marker and leaves the wrapper alone), and only
 * when it has an answer to give: a plugin install without its own `uuid` is
 * left alone, so ADK's failure is reported as it was. It does nothing where
 * Node's CommonJS loader is absent: the Workflow bundle resolves `node:module`
 * to an inert shim, and the hook has no role in the sandbox.
 *
 * The ES module loader is not hooked: it does not consult
 * `Module._resolveFilename`, and the API that reaches both loaders
 * (`module.registerHooks`, Node 22.15 and later) is still experimental, so an
 * ES module of the application that imports `@google/adk` itself still needs
 * `uuid` reachable from ADK's directory (the README documents the layout fix).
 * The plugin never loads ADK that way; its own imports are CommonJS, which
 * ADK's `exports` map serves from `dist/cjs`.
 */

import Module, { createRequire } from 'node:module';

/** The package ADK imports without declaring; the plugin's own dependency answers it. */
const UUID_PACKAGE = 'uuid';

/** A module file inside an installed `@google/adk` package, on either path separator. */
const ADK_MODULE_PATTERN = /[\\/]node_modules[\\/]@google[\\/]adk[\\/]/;

/**
 * Marks the installed wrapper, so that a second copy of this module (two
 * plugin versions in one process) finds it and leaves it alone.
 */
const INSTALLED: unique symbol = Symbol.for('@temporalio/google-adk.adkUuidResolution');

/** The subset of a CommonJS `Module` the hook reads from the requesting module. */
interface RequestingModule {
  filename?: string | null;
}

/** Node's `Module._resolveFilename` signature (an internal, stable since Node 0.x). */
type ResolveFilename = (
  this: unknown,
  request: string,
  parent: RequestingModule | null | undefined,
  isMain: boolean,
  options?: unknown
) => string;

/** The loader surface the hook touches; `@types/node` does not declare the internal. */
interface CommonJsLoader {
  _resolveFilename?: ResolveFilename & { [INSTALLED]?: true };
}

function isModuleNotFound(err: unknown): boolean {
  return typeof err === 'object' && err !== null && (err as { code?: unknown }).code === 'MODULE_NOT_FOUND';
}

/**
 * The plugin's own `uuid` entry, or `undefined` when the install lacks it. It
 * is resolved through a `require` made for this file rather than a
 * `require.resolve(...)` call: webpack follows the latter when it bundles this
 * module for the Workflow sandbox and warns at every Worker start that it
 * cannot (`Critical dependency: the request of a dependency is an
 * expression`), while the former is left alone, and never runs there.
 */
function pluginUuidEntry(): string | undefined {
  try {
    return createRequire(__filename).resolve(UUID_PACKAGE);
  } catch {
    return undefined;
  }
}

/**
 * Wraps Node's CommonJS resolver so that ADK's `uuid` request resolves to the
 * plugin's copy when nothing else answers it. Idempotent; a no-op without
 * Node's loader, and without the plugin's own copy of `uuid`. Runs at module
 * load; exported for tests. @internal
 */
export function installAdkUuidResolution(): void {
  const loader = Module as unknown as CommonJsLoader;
  const original = loader._resolveFilename;
  if (typeof original !== 'function' || original[INSTALLED]) return;
  const uuidEntry = pluginUuidEntry();
  if (uuidEntry === undefined) return;
  const wrapped: ResolveFilename & { [INSTALLED]?: true } = function (this: unknown, request, parent, isMain, options) {
    try {
      return original.call(this, request, parent, isMain, options);
    } catch (err) {
      if (request !== UUID_PACKAGE || !isModuleNotFound(err) || !ADK_MODULE_PATTERN.test(parent?.filename ?? '')) {
        throw err;
      }
      return uuidEntry;
    }
  };
  wrapped[INSTALLED] = true;
  loader._resolveFilename = wrapped;
}

installAdkUuidResolution();
