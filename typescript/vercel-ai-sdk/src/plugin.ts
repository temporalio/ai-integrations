import type { ProviderV4 } from '@ai-sdk/provider';
import { SimplePlugin } from '@temporalio/plugin';
import type { BundleOptions, Worker, WorkerOptions } from '@temporalio/worker';
import type { Duration } from '@temporalio/common/lib/time';
import { createActivities, _evictMcpConnectionsForServer } from './activities';
import { refuseLaterReplacement, refuseReplacedActivities } from './activity-sources';
import type { McpClientFactories } from './mcp';

/**
 * Options for the AI SDK plugin
 *
 * @experimental The AI SDK plugin is an experimental feature; APIs may change without notice.
 */
export interface AiSdkPluginOptions {
  modelProvider: ProviderV4;

  /**
   * This object contains a mapping of server names to functions which create MCP clients.
   * Any TemporalMCPClient used in a workflow should have its associated servername listed in this object.
   */
  mcpClientFactories?: McpClientFactories;

  /**
   * How long an idle worker-process MCP connection is kept open (and reused across `listTools`/`callTool`
   * activity invocations) before it's closed. Accepts a millisecond number or duration string (e.g.
   * `'5 minutes'`), like `startToCloseTimeout`. Defaults to 5 minutes.
   *
   * Pass `0` to disable connection reuse and restore the original create-then-close-per-call behavior,
   * for MCP servers/transports that don't tolerate a reused or concurrent session.
   */
  mcpConnectionIdleTimeout?: Duration;
}

/**
 * A Temporal plugin that integrates AI SDK providers for use in workflows.
 * This plugin creates activities that allow workflows to invoke AI models.
 *
 * Workflow code should import from `@temporalio/vercel-ai-sdk/workflow` (not the package root) so the
 * workflow bundle doesn't pull in the worker-side activities and their disallowed imports.
 *
 * @experimental The AI SDK plugin is an experimental feature; APIs may change without notice.
 */
export class AiSdkPlugin extends SimplePlugin {
  constructor(options: AiSdkPluginOptions) {
    const mcpServerNames = Object.keys(options.mcpClientFactories ?? {});
    super({
      name: 'AiSDKPlugin',
      activities: createActivities(options.modelProvider, options.mcpClientFactories, {
        mcpConnectionIdleTimeout: options.mcpConnectionIdleTimeout,
      }),
      runContext: async (next) => {
        try {
          await next();
        } finally {
          // Close any worker-process MCP connections opened lazily by the listTools/callTool activities.
          for (const name of mcpServerNames) {
            await _evictMcpConnectionsForServer(name);
          }
        }
      },
    });
  }

  /**
   * Adds the model and MCP Activities to the Worker.
   *
   * @throws TypeError if the Worker already has a different Activity with one
   * of these names: a caller Activity, another integration's Activity, or an
   * `AiSdkPlugin` with a different `modelProvider`, MCP client factory or
   * `mcpConnectionIdleTimeout`.
   */
  override configureWorker(options: WorkerOptions): WorkerOptions {
    refuseReplacedActivities('AiSdkPlugin', options.activities, this.options.activities as object);
    return super.configureWorker(options);
  }

  /**
   * Runs the Worker.
   *
   * @throws TypeError, before the Worker polls, if a plugin applied after this
   * one replaced one of its Activities with a different one. `Worker.create`
   * applies the Worker's plugins in order, then the `NativeConnection`'s.
   */
  override async runWorker(worker: Worker, next: (worker: Worker) => Promise<void>): Promise<void> {
    await refuseLaterReplacement('AiSdkPlugin', worker, this.options.activities as object, next);
    return super.runWorker(worker, next);
  }

  override configureBundler(options: BundleOptions): BundleOptions {
    // Prepend a polyfill-installer module to the webpack `entry` array so it
    // evaluates before any other module in the Workflow bundle. Webpack 5
    // emits multi-entry scripts that execute in array order at script-load,
    // so this guarantees the Web-API globals (`TransformStream`, `Headers`,
    // ...) exist before `ai` is evaluated (it defines `class ... extends
    // TransformStream` at module scope), making the relative import order of
    // `@temporalio/vercel-ai-sdk/workflow` and `ai` in workflow code irrelevant.
    const polyfillPath = require.resolve('./preload-polyfills');
    const baseOptions = super.configureBundler(options);
    const existingHook = baseOptions.webpackConfigHook;
    return {
      ...baseOptions,
      webpackConfigHook: (config) => {
        const cfg = existingHook ? existingHook(config) : config;
        const existingEntry = cfg.entry;
        const entries = Array.isArray(existingEntry)
          ? existingEntry
          : existingEntry !== undefined
            ? [existingEntry as string]
            : [];
        return { ...cfg, entry: [polyfillPath, ...entries] };
      },
    };
  }
}
