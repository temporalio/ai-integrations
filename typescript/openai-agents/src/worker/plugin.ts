import { trace } from '@opentelemetry/api';
import type { ModelProvider } from '@openai/agents-core';
import { SimplePlugin } from '@temporalio/plugin';
import type { BundleOptions, InjectedSinks, ReplayWorkerOptions, Worker, WorkerOptions } from '@temporalio/worker';
import { OpenAIAgentsTraceClientInterceptor } from '../client/trace-interceptor';
import { AGENT_TRACING_SINK_NAME, type AgentTracingSinks } from '../common/agent-sink-types';
import { MCP_STATEFUL_SUFFIX } from '../common/mcp-types';
import type { SerializableModelActivityOptions } from '../common/model-activity-options';
import { isReplaySafeTracerProvider } from '../common/tracing-bridge';
import { createModelActivity } from './activities';
import { ensureActivityTracingProcessorRegistered } from './activity-tracing';
import { makeAgentTracingSink } from './agent-sink-bridge';
import type { HostedToolCredentialsResolver } from './hosted-tool-credentials';
import type { StatelessMCPServerProvider } from './mcp-provider';
import { OnHostSandboxClientProvider } from './on-host-sandbox-provider';
import type { ActivityFunction, SandboxClientProvider } from './sandbox-provider';
import { StatefulMCPServerProvider } from './stateful-mcp-provider';
import {
  OpenAIAgentsTraceActivityInboundInterceptor,
  OpenAIAgentsTraceNexusInboundInterceptor,
  type OpenAIAgentsTraceInterceptorOptions,
  type TracingApiKeyResolver,
} from './trace-interceptor';

export type MCPServerProvider = StatelessMCPServerProvider | StatefulMCPServerProvider;

type WebpackConfig = Parameters<NonNullable<BundleOptions['webpackConfigHook']>>[0];
type WebpackCompiler = Parameters<Extract<NonNullable<WebpackConfig['plugins']>[number], (...args: any[]) => void>>[0];

const AGENTS_CORE_REQUEST = /^@openai\/agents-core(?:\/|$)/;

/**
 * `@openai/agents-core` has a CommonJS build and an ES module build. The plugin's own code is
 * CommonJS, so it loads the CommonJS build. An `import` in Workflow code, such as a `.ts`
 * Workflow file, would load the ES module build too: a second copy of the package. The plugin's
 * `instanceof` checks miss objects from that copy, and its dynamic `import()` becomes an async
 * chunk that cannot load in the Workflow sandbox. Resolving every request for the package as a
 * CommonJS request keeps one copy in the bundle.
 */
function resolveAgentsCoreAsCommonJs(compiler: WebpackCompiler): void {
  compiler.hooks.normalModuleFactory.tap('OpenAIAgentsPlugin', (factory) => {
    factory.hooks.beforeResolve.tap('OpenAIAgentsPlugin', (data) => {
      if (data.dependencyType === 'esm' && AGENTS_CORE_REQUEST.test(data.request)) {
        data.dependencyType = 'commonjs';
      }
    });
  });
}

// What each Activity an OpenAIAgentsPlugin builds was built from: its name, then what it depends on.
// configureWorker lets one of these replace another with the same entries: that is the same
// instance, or one built from the same options, registered twice (e.g. on the NativeConnection
// and the Worker).
const activitySources = new WeakMap<object, readonly unknown[]>();

function sameSources(a: object, b: object): boolean {
  const x = activitySources.get(a);
  const y = activitySources.get(b);
  return x !== undefined && y !== undefined && x.length === y.length && x.every((v, i) => v === y[i]);
}

// Marks the options of an OnHostSandboxClientProvider's Worker, which a plugin on the NativeConnection
// would otherwise configure as a main Worker.
const ON_HOST_WORKER_OPTIONS = Symbol('OpenAIAgentsPlugin.onHostWorkerOptions');

// The running Workers of OnHostSandboxClientProviders, which start no on-host Workers of their own.
const onHostWorkers = new WeakSet<Worker>();

/** An `OnHostSandboxClientProvider` and the Activities its Worker registers. */
interface OnHostProvider {
  provider: OnHostSandboxClientProvider;
  activities: Record<string, ActivityFunction>;
}

/** A Worker's Activity slot setting. */
type ActivityConcurrency = Pick<WorkerOptions, 'tuner' | 'maxConcurrentActivityTaskExecutions'>;

function workerKey(namespace: string | undefined, taskQueue: string | undefined): string {
  return JSON.stringify([namespace ?? 'default', taskQueue]);
}

/**
 * Ends a Worker that has not started polling. There is no public way to stop one. With Core shut down
 * first, the pollers end at once and run()'s own teardown releases the connection.
 */
async function discardUnstarted(worker: Worker, run: (worker: Worker) => Promise<void>): Promise<void> {
  (worker as unknown as { nativeWorker: { initiateShutdown(): void } }).nativeWorker.initiateShutdown();
  await run(worker).catch(() => undefined);
}

/**
 * Options controlling trace interceptor behavior on both the Activity side
 * (directly) and the Workflow side (via header propagation).
 */
export interface OpenAIAgentsPluginInterceptorOptions {
  /** Wrap calls in `temporal:*` custom spans. @default false */
  addTemporalSpans?: boolean;
  /**
   * Emit OTel spans for agent-SDK trace/span events. Orthogonal to
   * `addTemporalSpans`. @default false
   */
  useOtelInstrumentation?: boolean;
}

export interface OpenAIAgentsPluginOptions {
  /** Model provider used to resolve model names to Model instances. */
  modelProvider: ModelProvider;
  /** MCP server providers whose Activities will be auto-registered. */
  mcpServerProviders?: MCPServerProvider[];
  /**
   * Sandbox client providers whose Activities will be auto-registered.
   * Reference them from Workflows via `temporalSandboxClient(name)`.
   *
   * The Worker also runs an extra Worker, on the same connection, for each
   * `OnHostSandboxClientProvider`: it polls the Task Queue
   * `<Task Queue>/<provider name>@<workerName>`, starts when the Worker runs and stops with it,
   * and takes the Worker's data converter, Activity interceptors, Activity concurrency, Activity
   * rate limits and shutdown grace time. No plugin runs on it, so with an on-host provider the
   * plugin refuses a Worker whose `NativeConnection` has a plugin other than `OpenAIAgentsPlugin`:
   * register that plugin on the Worker instead.
   *
   * @experimental Sandbox support is experimental and may change without notice.
   */
  sandboxClientProviders?: Array<SandboxClientProvider | OnHostSandboxClientProvider>;
  /**
   * Resolves credentials for the hosted tools an Agent declares — a hosted MCP
   * tool's `authorization` and `headers`, a shell or code interpreter tool's
   * domain secrets — so Workflow code can omit them and keep them out of
   * Workflow history. See {@link HostedToolCredentialsResolver}.
   *
   * @experimental Worker-side hosted tool credentials are experimental and may change without notice.
   */
  hostedToolCredentials?: HostedToolCredentialsResolver;
  /**
   * Allowlist of Worker environment variable names this Worker will read on
   * behalf of a `workerEnvValue` in a sandbox `Manifest`. Names must match
   * exactly, except for the entry `'*'`, which allows every name.
   *
   * An allowlisted variable that is unset or empty reads as the empty string. A
   * name outside the allowlist fails the Activity non-retryably.
   *
   * @default [] — no name is readable.
   * @experimental Worker environment variable references are experimental and may change without notice.
   */
  resolvableWorkerEnvVars?: readonly string[];
  /**
   * Default Model Activity options (timeouts, retry, Task Queue, etc.).
   * Propagated to the Workflow via the `__openai_agents_config` header.
   */
  modelParams?: SerializableModelActivityOptions;
  /**
   * Trace interceptor options. Propagated to the Workflow via the
   * `__openai_agents_config` header and applied to Activities directly.
   */
  interceptorOptions?: OpenAIAgentsPluginInterceptorOptions;
  /**
   * Resolves the tracing export API key for spans that cross a Temporal
   * boundary, on the Worker, so the key stays out of Workflow history. Use it
   * when runs export traces with different keys. See {@link TracingApiKeyResolver}.
   */
  tracingApiKey?: TracingApiKeyResolver;
}

/** Reads this package's version, which worker heartbeats report. Empty if it can't be read. */
function packageVersion(): string {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    return (require('../../package.json') as { version?: string }).version ?? '';
  } catch {
    return '';
  }
}

/**
 * Temporal plugin integrating the OpenAI Agents SDK. Registers model
 * invocation Activities so Workflow-side ActivityBackedModel can delegate LLM
 * calls, and wires trace propagation between client/Workflow/Activity.
 */
export class OpenAIAgentsPlugin extends SimplePlugin {
  /** The package version, reported in worker heartbeats. */
  readonly version = packageVersion();

  // The provider each provider Activity comes from, by Activity name, for the error message.
  private readonly activityProviders: ReadonlyMap<string, string>;
  private readonly tracingApiKey?: TracingApiKeyResolver;
  private readonly onHostProviders: readonly OnHostProvider[];
  // The Activity slot setting of each main Worker this plugin configured, by namespace and Task Queue:
  // the compiled options of a running Worker keep only the native form of its tuner.
  private readonly activityConcurrency = new Map<string, ActivityConcurrency>();

  constructor(options: OpenAIAgentsPluginOptions) {
    const modelActivities = createModelActivity(options.modelProvider, options.hostedToolCredentials);
    for (const [name, fn] of Object.entries(modelActivities)) {
      activitySources.set(fn, [name, options.modelProvider, options.hostedToolCredentials]);
    }

    let allActivities: Record<string, (...args: any[]) => Promise<any>> = { ...modelActivities };

    const interceptorOpts = options.interceptorOptions;
    for (const provider of options.sandboxClientProviders ?? []) {
      provider._setResolvableWorkerEnvVars(options.resolvableWorkerEnvVars ?? []);
      provider._addTemporalSpans = interceptorOpts?.addTemporalSpans === true;
    }

    const providers: Array<{ name: string; _getActivities(): Record<string, (...args: any[]) => Promise<any>> }> = [
      ...(options.mcpServerProviders ?? []),
      ...(options.sandboxClientProviders ?? []),
    ];
    const seenNames = new Set<string>();
    const activityProviders = new Map<string, string>();
    const onHostProviders: OnHostProvider[] = [];
    for (const provider of providers) {
      if (seenNames.has(provider.name)) {
        throw new Error(
          `Duplicate provider name: '${provider.name}'. Each MCP server and sandbox client provider must have a unique name.`
        );
      }
      seenNames.add(provider.name);
      // A stateful MCP server provider adds a suffix to the name the caller passed.
      const shownName =
        provider instanceof StatefulMCPServerProvider ? provider.name.slice(0, -MCP_STATEFUL_SUFFIX.length) : provider.name;
      const providerActivities = provider._getActivities();
      for (const [name, fn] of Object.entries(providerActivities)) {
        activitySources.set(fn, [name, provider]);
        activityProviders.set(name, shownName);
      }
      if (provider instanceof OnHostSandboxClientProvider) {
        onHostProviders.push({ provider, activities: providerActivities });
      }
      allActivities = { ...allActivities, ...providerActivities };
    }

    const activityInterceptorOptions: OpenAIAgentsTraceInterceptorOptions = {
      addTemporalSpans: interceptorOpts?.addTemporalSpans,
      tracingApiKey: options.tracingApiKey,
    };

    if (interceptorOpts?.useOtelInstrumentation) {
      // OTel's `setGlobalTracerProvider` wraps the caller's provider in a ProxyTracerProvider;
      // unwrap to read the brand off the actual delegate.
      const globalProvider = trace.getTracerProvider();
      const actualProvider = (globalProvider as { getDelegate?: () => unknown }).getDelegate?.() ?? globalProvider;
      if (!isReplaySafeTracerProvider(actualProvider)) {
        throw new Error(
          'Global TracerProvider must be created via createTracerProvider from @temporalio/openai-agents/otel. ' +
            'Call setGlobalTracerProvider(createTracerProvider({...})) before constructing OpenAIAgentsPlugin.'
        );
      }
    }

    ensureActivityTracingProcessorRegistered();

    super({
      name: 'OpenAIAgentsPlugin',
      activities: allActivities,
      clientInterceptors: {
        workflow: [
          new OpenAIAgentsTraceClientInterceptor({
            addTemporalSpans: interceptorOpts?.addTemporalSpans,
            useOtelInstrumentation: interceptorOpts?.useOtelInstrumentation,
            modelParams: options.modelParams,
          }),
        ],
      },
      workerInterceptors: {
        workflowModules: [require.resolve('../workflow/trace-interceptor')],
        activity: [
          (ctx) => ({ inbound: new OpenAIAgentsTraceActivityInboundInterceptor(ctx, activityInterceptorOptions) }),
        ],
        nexus: [
          () => ({ inbound: new OpenAIAgentsTraceNexusInboundInterceptor({ tracingApiKey: options.tracingApiKey }) }),
        ],
      },
    });
    this.activityProviders = activityProviders;
    this.tracingApiKey = options.tracingApiKey;
    this.onHostProviders = onHostProviders;
  }

  /**
   * Adds the plugin's Activities, interceptors and sinks to the Worker.
   *
   * @throws TypeError if the Worker already has a different Activity with one of these names: a
   * caller Activity, another integration's Activity, or one from an `OpenAIAgentsPlugin` built from
   * a different model provider, credentials resolver, or MCP server or sandbox client provider.
   * Activities that a plugin applied later replaces are refused by {@link runWorker}.
   */
  override configureWorker(options: WorkerOptions): WorkerOptions {
    // The Worker of an OnHostSandboxClientProvider gets what it needs from its main Worker.
    if ((options as { [ON_HOST_WORKER_OPTIONS]?: boolean })[ON_HOST_WORKER_OPTIONS] === true) return options;
    if (this.onHostProviders.length > 0) {
      // Worker.create applies the NativeConnection's plugins to the extra Workers too.
      const other = options.connection?.plugins.find((plugin) => !(plugin instanceof OpenAIAgentsPlugin));
      if (other !== undefined) {
        throw new TypeError(
          `Plugin '${other.name}' is on the NativeConnection, so it would also run on the extra Worker of ` +
            "OpenAIAgentsPlugin's OnHostSandboxClientProvider. Register it on the Worker (Worker.create " +
            'plugins) instead of the NativeConnection.'
        );
      }
      this.activityConcurrency.set(workerKey(options.namespace, options.taskQueue), {
        tuner: options.tuner,
        maxConcurrentActivityTaskExecutions: options.maxConcurrentActivityTaskExecutions,
      });
    }
    const existing = options.activities as Record<string, unknown> | undefined;
    for (const [name, fn] of Object.entries(this.options.activities as object)) {
      const other = existing && Object.prototype.propertyIsEnumerable.call(existing, name) ? existing[name] : undefined;
      if (typeof other === 'function' && other !== fn && !sameSources(other, fn)) {
        const provider = this.activityProviders.get(name);
        const source = provider === undefined ? 'OpenAIAgentsPlugin' : `OpenAIAgentsPlugin provider '${provider}'`;
        throw new TypeError(
          `Duplicate registration of Activity '${name}': ${source} would replace a different Activity with this ` +
            `name. ${this.remedy(name)}`
        );
      }
    }
    return super.configureWorker(this.injectSinks(options));
  }

  /**
   * Runs the Worker, and with it a Worker for each `OnHostSandboxClientProvider`. When one of them
   * stops, the others are shut down.
   *
   * @throws TypeError, before the Worker polls, if a plugin applied after this one replaced one of
   * its Activities with a different one. `Worker.create` applies the Worker's plugins in order, then
   * the `NativeConnection`'s.
   */
  override async runWorker(worker: Worker, next: (worker: Worker) => Promise<void>): Promise<void> {
    if (onHostWorkers.has(worker)) return next(worker);
    for (const [name, fn] of Object.entries(this.options.activities as object)) {
      const kept = worker.options.activities.get(name);
      if (kept === undefined || kept === fn || sameSources(kept, fn)) continue;
      await discardUnstarted(worker, next);
      throw new TypeError(
        `Duplicate registration of Activity '${name}': a plugin applied after OpenAIAgentsPlugin replaced its ` +
          `Activity with this name. ${this.remedy(name)}`
      );
    }
    return super.runWorker(worker, (main) => this.runWithOnHostWorkers(main, next));
  }

  private async runWithOnHostWorkers(worker: Worker, next: (worker: Worker) => Promise<void>): Promise<void> {
    if (this.onHostProviders.length === 0 || worker.connection === undefined) return next(worker);
    const { Worker: WorkerClass } = await import('@temporalio/worker');
    const { namespace, taskQueue } = worker.options;
    const concurrency = this.activityConcurrency.get(workerKey(namespace, taskQueue));
    this.activityConcurrency.delete(workerKey(namespace, taskQueue));
    const onHost: Worker[] = [];
    try {
      for (const { provider, activities } of this.onHostProviders) {
        const extraTaskQueue = provider._taskQueueFor(taskQueue);
        // Its sandbox operations ran on the main Worker before, so it keeps the main Worker's Activity
        // setup.
        const options: WorkerOptions = {
          // Without a connection from the user, each Worker opens and closes its own: a Worker can't close
          // the one it opened while another Worker still uses it. No Worker closes the user's connection.
          connection: worker.options.connection,
          namespace,
          taskQueue: extraTaskQueue,
          activities,
          dataConverter: worker.options.dataConverter,
          interceptors: { activity: worker.options.interceptors.activity },
          ...concurrency,
          maxActivitiesPerSecond: worker.options.maxActivitiesPerSecond,
          maxTaskQueueActivitiesPerSecond: worker.options.maxTaskQueueActivitiesPerSecond,
          // In milliseconds: from @temporalio/worker 1.26, TypeScript can't check the compiled
          // `shutdownGraceTime` type against WorkerOptions (TS2859).
          shutdownGraceTime: worker.options.shutdownGraceTimeMs,
          shutdownForceTime: worker.options.shutdownForceTime,
        };
        let created: Worker;
        try {
          created = await WorkerClass.create(Object.assign(options, { [ON_HOST_WORKER_OPTIONS]: true }));
        } catch (err) {
          // Core refuses a second Worker on the queue, such as when another plugin on this Worker has an
          // on-host provider with the same name and Worker name, or another Worker in this process polls
          // the same main Task Queue.
          throw new Error(
            `Cannot start the Worker of OnHostSandboxClientProvider '${provider.name}' on Task Queue ` +
              `'${extraTaskQueue}': ${err instanceof Error ? err.message : String(err)}`,
            { cause: err }
          );
        }
        onHostWorkers.add(created);
        onHost.push(created);
      }
    } catch (err) {
      await Promise.all([
        discardUnstarted(worker, next),
        ...onHost.map((created) => discardUnstarted(created, (w) => w.run())),
      ]);
      throw err;
    }
    const all = [worker, ...onHost];
    const stopAll = () => {
      for (const w of all) if (w.getState() === 'RUNNING') w.shutdown();
    };
    const results = await Promise.allSettled(
      [next(worker), ...onHost.map((w) => w.run())].map((run) => run.finally(stopAll))
    );
    const failed = results.find((result): result is PromiseRejectedResult => result.status === 'rejected');
    if (failed) throw failed.reason;
  }

  override configureReplayWorker(options: ReplayWorkerOptions): ReplayWorkerOptions {
    return super.configureReplayWorker(this.injectSinks(options));
  }

  override configureBundler(options: BundleOptions): BundleOptions {
    // Prepend a polyfill-installer module to the webpack `entry` array so it
    // evaluates before any other module in the Workflow bundle. Webpack 5
    // emits multi-entry scripts that execute in array order at script-load,
    // so this guarantees `globalThis.ReadableStream` (and the other Web-API
    // shims) are populated before agents-core's browser shim captures them
    // into module-level constants.
    const polyfillPath = require.resolve('../workflow/preload-polyfills');
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
        const plugins = [...(cfg.plugins ?? []), resolveAgentsCoreAsCommonJs];
        return { ...cfg, entry: [polyfillPath, ...entries], plugins };
      },
    };
  }

  private remedy(name: string): string {
    const rename = this.activityProviders.has(name) ? 'Rename the provider, rename' : 'Rename';
    return (
      `${rename} or remove the other Activity, run the other integration on its own Worker and task queue, ` +
      'or use one OpenAIAgentsPlugin instance on this Worker.'
    );
  }

  private injectSinks<T extends { sinks?: InjectedSinks<any> }>(options: T): T {
    // Registered unconditionally; per-Workflow useOtelInstrumentation flag drives suppression
    // at the bridge layer via the SUPPRESS_OTEL_BRIDGE marker.
    const sinks: InjectedSinks<AgentTracingSinks> = {
      [AGENT_TRACING_SINK_NAME]: makeAgentTracingSink(this.tracingApiKey),
    };
    return {
      ...options,
      sinks: {
        ...options.sinks,
        ...sinks,
      },
    };
  }
}
