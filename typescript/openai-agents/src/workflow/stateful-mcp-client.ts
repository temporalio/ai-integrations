import type { MCPToolFilterCallable, MCPToolFilterStatic } from '@openai/agents-core';
import {
  ActivityFailure,
  ApplicationFailure,
  TimeoutFailure,
  TimeoutType,
  type Duration,
  type RetryPolicy,
} from '@temporalio/common';
import { msOptionalToNumber } from '@temporalio/common/lib/time';
import {
  type ActivityOptions,
  CancellationScope,
  isCancellation,
  scheduleActivity,
  workflowInfo,
} from '@temporalio/workflow';
import {
  DEDICATED_WORKER_FAILURE_TYPE,
  MCP_CALL_TOOL_SUFFIX,
  MCP_LIST_TOOLS_SUFFIX,
  MCP_SERVER_SESSION_SUFFIX,
  MCP_STATEFUL_SUFFIX,
  type McpServerHandleOptions,
  type StatefulMcpServerSessionArgs,
  type StatefulTemporalMCPServer,
} from '../common/mcp-types';
import { getCurrentPluginConfig } from './plugin-config-store';

export interface StatefulMcpServerOptions extends McpServerHandleOptions {
  /** Activity options for tool operation Activities on the dedicated Worker. */
  config?: {
    startToCloseTimeout?: Duration;
    scheduleToStartTimeout?: Duration;
    heartbeatTimeout?: Duration;
    taskQueue?: string;
    retryPolicy?: RetryPolicy;
  };
  /** Activity options for the long-running server-session Activity. */
  serverSessionConfig?: {
    startToCloseTimeout?: Duration;
    heartbeatTimeout?: Duration;
  };
  factoryArgument?: unknown;
  /**
   * When `true` (the default), the tool list is fetched once and reused for
   * the rest of the Workflow run. Set to `false` if the server's tool set may
   * change mid-run.
   */
  cacheToolsList?: boolean;
}

const DEDICATED_WORKER_SCHEDULE_FAILURE_MESSAGE =
  'MCP Stateful Server Worker failed to schedule activity. Check that a worker is polling the dedicated task queue for this MCP run.';

const DEDICATED_WORKER_HEARTBEAT_FAILURE_MESSAGE =
  'MCP Stateful Server Worker failed to heartbeat. Check that a worker is polling the dedicated task queue for this MCP run.';

async function handleWorkerFailure<T>(fn: () => Promise<T>): Promise<T> {
  try {
    return await fn();
  } catch (err: unknown) {
    if (err instanceof ActivityFailure && err.cause instanceof TimeoutFailure) {
      if (err.cause.timeoutType === TimeoutType.SCHEDULE_TO_START) {
        throw ApplicationFailure.create({
          message: DEDICATED_WORKER_SCHEDULE_FAILURE_MESSAGE,
          type: DEDICATED_WORKER_FAILURE_TYPE,
          nonRetryable: false,
        });
      }
      if (err.cause.timeoutType === TimeoutType.HEARTBEAT) {
        throw ApplicationFailure.create({
          message: DEDICATED_WORKER_HEARTBEAT_FAILURE_MESSAGE,
          type: DEDICATED_WORKER_FAILURE_TYPE,
          nonRetryable: false,
        });
      }
    }
    throw err;
  }
}

/**
 * Workflow-side handle for a stateful MCP server connection. Tool operations
 * route to a dedicated in-process Worker on a per-run Task Queue, preserving
 * server state across calls.
 */
export class StatefulMCPServerReference implements StatefulTemporalMCPServer {
  private readonly _name: string;
  private readonly _operationConfig: ActivityOptions;
  private readonly _sessionConfig: ActivityOptions;
  private readonly _factoryArgument: unknown;
  private readonly _cacheToolsList: boolean;
  private _sessionScope: CancellationScope | undefined;
  private _sessionPromise: Promise<void> | undefined;
  private _sessionEnded: Promise<void> | undefined;
  private _sessionSettled = false;
  private _sessionError: unknown;
  private _connected = false;
  private _cachedTools: any[] | undefined;
  toolFilter?: MCPToolFilterCallable | MCPToolFilterStatic;

  constructor(name: string, options?: StatefulMcpServerOptions) {
    this._name = `${name}${MCP_STATEFUL_SUFFIX}`;
    this._operationConfig = {
      startToCloseTimeout: options?.config?.startToCloseTimeout ?? '1 minute',
      scheduleToStartTimeout: options?.config?.scheduleToStartTimeout ?? '30 seconds',
      heartbeatTimeout: options?.config?.heartbeatTimeout,
      taskQueue: options?.config?.taskQueue,
      retry: options?.config?.retryPolicy,
    };
    this._sessionConfig = {
      startToCloseTimeout: options?.serverSessionConfig?.startToCloseTimeout ?? '1 hour',
      heartbeatTimeout: options?.serverSessionConfig?.heartbeatTimeout,
    };
    this._factoryArgument = options?.factoryArgument;
    this._cacheToolsList = options?.cacheToolsList ?? true;
    // A callable filter gets the registered name, as with statelessMcpServer(), not `this.name`.
    const toolFilter = options?.toolFilter;
    this.toolFilter =
      typeof toolFilter === 'function'
        ? (context, tool) => toolFilter({ ...context, serverName: name }, tool)
        : toolFilter;
  }

  get name(): string {
    return this._name;
  }

  // The agent SDK caches the filtered list, so with a filter only this handle caches, and the
  // agent filters every listing.
  get cacheToolsList(): boolean {
    return this._cacheToolsList && !this.toolFilter;
  }

  /** Starts the per-run server-session Activity. */
  async connect(): Promise<void> {
    const runId = workflowInfo().runId;
    this._operationConfig.taskQueue = `${this._name}@${runId}`;

    const sessionActivityName = `${this._name}${MCP_SERVER_SESSION_SUFFIX}`;

    // Forward this Workflow's trace-interceptor settings to the session
    // Activity so the dedicated per-run Worker wires its Activity interceptor
    // identically — otherwise tool Activities would skip trace-context
    // restoration and the `temporal:executeActivity:*` span.
    const pluginConfig = getCurrentPluginConfig();
    const args: StatefulMcpServerSessionArgs = {
      factoryArgument: this._factoryArgument,
      activityInterceptorOptions: pluginConfig ? { addTemporalSpans: pluginConfig.addTemporalSpans } : undefined,
    };

    this._sessionScope = new CancellationScope();
    this._sessionPromise = this._sessionScope.run(() =>
      scheduleActivity<void>(sessionActivityName, [args], this._sessionConfig)
    );
    // Errors are observed during cleanup, not here.
    this._sessionPromise.catch(() => {});
    this._sessionSettled = false;
    this._sessionError = undefined;
    this._sessionEnded = this._sessionPromise.then(
      () => {
        this._sessionSettled = true;
      },
      (err: unknown) => {
        this._sessionSettled = true;
        this._sessionError = err;
      }
    );
    this._connected = true;
  }

  /** Cancels the server-session Activity, tearing down the dedicated Worker. */
  async cleanup(): Promise<void> {
    if (this._sessionScope) {
      this._sessionScope.cancel();
      try {
        await this._sessionPromise;
      } catch (err: unknown) {
        if (!isCancellation(err)) {
          throw err;
        }
      }
      this._sessionScope = undefined;
      this._sessionPromise = undefined;
      this._connected = false;
    }
  }

  /** Alias for `cleanup()` — matches the MCPServer interface. */
  async close(): Promise<void> {
    await this.cleanup();
  }

  private assertConnected(): void {
    if (!this._connected) {
      throw ApplicationFailure.create({
        message: 'Stateful MCP Server not connected — did you forget to call `connect()` before using the server?',
      });
    }
  }

  private sessionEndedFailure(): ApplicationFailure {
    return ApplicationFailure.create({
      message: "The stateful MCP server session Activity has ended, so no Worker serves this server's operations.",
      type: DEDICATED_WORKER_FAILURE_TYPE,
      cause: this._sessionError instanceof Error ? this._sessionError : undefined,
    });
  }

  /**
   * Runs an operation Activity on the session's Task Queue. Only the session
   * Activity's Worker serves that queue. A schedule-to-start timeout already
   * fails an operation once that Worker is gone. Without one, the operation
   * would wait forever, so fail it with `DedicatedWorkerFailure` once the
   * session Activity has ended.
   */
  private async executeOperation<R>(activity: string, args: unknown[]): Promise<R> {
    if (msOptionalToNumber(this._operationConfig.scheduleToStartTimeout)) {
      return scheduleActivity<R>(activity, args, this._operationConfig);
    }
    if (this._sessionSettled) {
      throw this.sessionEndedFailure();
    }
    // A child of the caller's scope, so the caller's cancellation still reaches the operation.
    const scope = new CancellationScope();
    const operation = scope.run(() => scheduleActivity<R>(activity, args, this._operationConfig));
    const first = await Promise.race([
      operation.then(
        (value) => ({ done: true as const, value }),
        (error: unknown) => ({ done: true as const, error })
      ),
      this._sessionEnded!.then(() => ({ done: false as const })),
    ]);
    if (first.done) {
      if ('error' in first) throw first.error;
      return first.value;
    }
    operation.catch(() => {});
    scope.cancel();
    throw this.sessionEndedFailure();
  }

  async listTools(): Promise<any[]> {
    this.assertConnected();
    const shouldCache = this._cacheToolsList && !!this.toolFilter;
    if (shouldCache && this._cachedTools) {
      return this._cachedTools;
    }
    const tools = await handleWorkerFailure(() => this.executeOperation<any[]>(this._name + MCP_LIST_TOOLS_SUFFIX, []));
    if (shouldCache) {
      this._cachedTools = tools;
    }
    return tools;
  }

  async callTool(toolName: string, args: Record<string, unknown> | null): Promise<any> {
    this.assertConnected();
    return handleWorkerFailure(() =>
      this.executeOperation(this._name + MCP_CALL_TOOL_SUFFIX, [{ toolName, args }])
    );
  }

  async invalidateToolsCache(): Promise<void> {
    // Without a filter, caching is owned by the upstream agent SDK. With one, this cache holds the
    // full list, and the agent SDK's invalidateServerToolsCache(name) cannot clear it.
    this._cachedTools = undefined;
  }
}

/**
 * Creates a stateful MCP server handle. Maintains a persistent MCP connection
 * across the Workflow via a dedicated per-run Worker.
 *
 * If the dedicated Worker fails (startup timeout or missed heartbeat), or the
 * session Activity has ended and `config.scheduleToStartTimeout` is `0`,
 * operations throw `ApplicationFailure` with type `"DedicatedWorkerFailure"`.
 *
 * @param name - Server name; must match the `StatefulMCPServerProvider` registered on the Worker side.
 */
export function statefulMcpServer(name: string, options?: StatefulMcpServerOptions): StatefulTemporalMCPServer {
  return new StatefulMCPServerReference(name, options);
}
