/**
 * `LangGraphPlugin`: registers the Activities for a {@link LangGraphDefinitions}
 * and makes LangGraph.js run in the Workflow sandbox.
 *
 * @module
 */

import { SimplePlugin } from '@temporalio/plugin';
import type { BundleOptions, Worker, WorkerOptions } from '@temporalio/worker';
import { createActivities } from './activities';
import { refuseLaterReplacement, refuseReplacedActivities } from './activity-sources';
import { internalsOf } from './definitions';
import type { LangGraphDefinitions } from './definitions';

type WebpackConfig = Parameters<NonNullable<BundleOptions['webpackConfigHook']>>[0];

interface CompilerLike {
  webpack: {
    NormalModuleReplacementPlugin: new (
      pattern: RegExp,
      replace: (resource: { request: string }) => void
    ) => { apply(compiler: CompilerLike): void };
  };
}

const ASYNC_HOOKS = /^(node:)?async_hooks$/;
const LANGGRAPH_DIST = /[\\/]@langchain[\\/]langgraph[\\/]dist[\\/].*\.c?js$/;

/** Reads this package's version, which Worker heartbeats report. Empty if it can't be read. */
function packageVersion(): string {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    return (require('../package.json') as { version?: string }).version ?? '';
  } catch {
    return '';
  }
}

/** Adds the plugin's module replacement and loader after any existing hook. */
function sandboxCompat(
  existing: BundleOptions['webpackConfigHook']
): NonNullable<BundleOptions['webpackConfigHook']> {
  const shim = require.resolve('./async-hooks-shim');
  const loader = require.resolve('./stack-trace-limit-loader');
  return (config: WebpackConfig): WebpackConfig => {
    const cfg = existing ? existing(config) : config;
    type PluginElement = NonNullable<WebpackConfig['plugins']>[number];
    const replaceAsyncHooks = {
      apply(compiler: CompilerLike) {
        new compiler.webpack.NormalModuleReplacementPlugin(ASYNC_HOOKS, (resource) => {
          resource.request = shim;
        }).apply(compiler);
      },
    };
    cfg.plugins = [...(cfg.plugins ?? []), replaceAsyncHooks as unknown as PluginElement];
    type Rule = NonNullable<NonNullable<WebpackConfig['module']>['rules']>[number];
    const rule = { test: LANGGRAPH_DIST, use: [{ loader }] } as Rule;
    cfg.module = { ...cfg.module, rules: [...(cfg.module?.rules ?? []), rule] };
    return cfg;
  };
}

/**
 * The Temporal plugin for LangGraph.js. It runs the nodes and tasks of a
 * {@link LangGraphDefinitions} as Activities or inline in the Workflow, and
 * configures the Workflow bundle for LangGraph.
 *
 * Register it on the Worker:
 *
 * ```ts
 * const worker = await Worker.create({
 *   workflowsPath: require.resolve('./workflows'),
 *   plugins: [new LangGraphPlugin(langGraph)],
 *   // ...
 * });
 * ```
 *
 * A Worker that takes a prebuilt `workflowBundle` must build it with
 * `bundleWorkflowCode({ plugins: [plugin], ... })`.
 *
 * @experimental The LangGraph plugin is an experimental feature; APIs may change without notice.
 */
export class LangGraphPlugin extends SimplePlugin {
  /** The package version, reported in Worker heartbeats. */
  readonly version = packageVersion();

  /**
   * @param definitions The graphs, entrypoints and tasks, from `defineLangGraph()`.
   */
  constructor(definitions: LangGraphDefinitions<any, any>) {
    super({
      name: 'langchain.LangGraphPlugin',
      activities: createActivities(internalsOf(definitions)),
    });
  }

  /**
   * Adds the Activities of the definitions' Activity nodes and tasks to the
   * Worker.
   *
   * @throws TypeError if the Worker already has a different Activity with one
   * of these names: a caller Activity, another integration's Activity, or a
   * `LangGraphPlugin` whose definitions run a different function or stream
   * differently under that name.
   */
  override configureWorker(options: WorkerOptions): WorkerOptions {
    refuseReplacedActivities('LangGraphPlugin', options.activities, this.options.activities as object);
    return super.configureWorker(options);
  }

  /**
   * Runs the Worker.
   *
   * @throws TypeError, before the Worker polls, if a plugin applied after this
   * one replaced one of its Activities with a different one. `Worker.create`
   * applies the Worker's plugins in order, then the `NativeConnection`'s.
   */
  override runWorker(worker: Worker, next: (worker: Worker) => Promise<void>): Promise<void> {
    // Not async: when the check passes, the Worker starts in this tick, as
    // `Worker.runUntil` needs (see `refuseLaterReplacement`).
    return (
      refuseLaterReplacement('LangGraphPlugin', worker, this.options.activities as object, next) ??
      super.runWorker(worker, next)
    );
  }

  /**
   * Adds the plugin's Workflow interceptor modules, the polyfills first so
   * they precede any module that loads LangChain, and makes LangGraph's
   * modules load in the sandbox.
   */
  override configureBundler(options: BundleOptions): BundleOptions {
    const configured = super.configureBundler(options);
    return {
      ...configured,
      workflowInterceptorModules: [
        require.resolve('./load-polyfills'),
        ...(configured.workflowInterceptorModules ?? []),
        require.resolve('./workflow-interceptors'),
      ],
      webpackConfigHook: sandboxCompat(configured.webpackConfigHook),
    };
  }
}
