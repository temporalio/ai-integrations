import type { MCPServer } from '@openai/agents-core';
import {
  ApplicationFailure,
  CancelledFailure,
  NamespaceNotFoundError,
  WorkflowNotFoundError,
  type SerializationContext,
} from '@temporalio/common';
import { Context, activityInfo } from '@temporalio/activity';
import type { NativeConnection } from '@temporalio/worker';
import {
  DEDICATED_WORKER_FAILURE_TYPE,
  MCP_CALL_TOOL_SUFFIX,
  MCP_LIST_TOOLS_SUFFIX,
  MCP_SERVER_SESSION_SUFFIX,
  MCP_STATEFUL_SUFFIX,
  type StatefulMcpServerSessionArgs,
} from '../common/mcp-types';
import { startAdaptiveHeartbeat } from './heartbeat';
import { assertMCPServer, callToolWithMeta, filterListedTools, rejectCallableToolFilter, rejectToolMetaResolver } from './mcp-provider';

export function registerCancelShutdown(ctx: Context, dedicatedWorker: { shutdown: () => void }): void {
  ctx.cancelled.catch(() => {
    dedicatedWorker.shutdown();
  });
}

interface CallToolArgs {
  toolName: string;
  args: Record<string, unknown> | null;
  meta?: Record<string, unknown>;
}

// Heartbeat details of a session Activity attempt that connected its MCP
// server. Heartbeat details carry over to the next attempt, which must not open
// a new, empty session in its place.
const SESSION_OPENED = 'mcp-session-opened';
// Heartbeat details of an attempt that has not connected yet. The Worker sends
// heartbeats later, so an attempt can be lost after connecting but before its
// marker reaches the server.
const SESSION_MAY_BE_OPEN = 'mcp-session-may-be-open';

// How long the session Activity waits for the server to show its first heartbeat.
const STORE_TIMEOUT_MS = 30_000;

// gRPC status codes of describe errors that retrying cannot fix.
const DESCRIBE_REFUSALS = new Map([
  [3, 'INVALID_ARGUMENT'],
  [5, 'NOT_FOUND'],
  [7, 'PERMISSION_DENIED'],
  [12, 'UNIMPLEMENTED'],
  [16, 'UNAUTHENTICATED'],
]);

// Describe errors that mean the Worker's client lacks permission.
const DESCRIBE_PERMISSION_ERRORS = new Set(['PERMISSION_DENIED', 'UNAUTHENTICATED']);

/** The status of a describe error that retrying cannot fix, or undefined. */
function describeRefusal(err: unknown): { status: string; details: string } | undefined {
  if (err instanceof WorkflowNotFoundError || err instanceof NamespaceNotFoundError) {
    return { status: 'NOT_FOUND', details: err.message };
  }
  // The client wraps other gRPC errors in a ServiceError.
  const cause = (err as { cause?: { code?: unknown; details?: unknown } } | undefined)?.cause;
  const status = typeof cause?.code === 'number' ? DESCRIBE_REFUSALS.get(cause.code) : undefined;
  return status ? { status, details: String(cause?.details) } : undefined;
}

/** The failure for a retry whose earlier attempt opened, or may have opened, a session. */
function earlierAttemptFailure(details: unknown): ApplicationFailure | undefined {
  let reason: string;
  if (details === SESSION_OPENED) {
    reason = 'opened an MCP session';
  } else if (details === SESSION_MAY_BE_OPEN) {
    reason = 'ended (it lost its Worker, timed out, or was reset) without confirming that it opened no MCP session';
  } else {
    return undefined;
  }
  return ApplicationFailure.create({
    message: `An earlier attempt of this stateful MCP server session Activity ${reason}, and Temporal cannot recreate an MCP session's state.`,
    type: DEDICATED_WORKER_FAILURE_TYPE,
    nonRetryable: true,
  });
}

/**
 * Heartbeat details of one session Activity attempt.
 *
 * Only the Worker sends them, one at a time and in order, and it sends the
 * latest ones before the attempt's failure (not before a cancellation). The plugin cannot wait for a
 * heartbeat, so every heartbeat says a session may be open until the attempt
 * ends without one, and the server is not connected until the Temporal server
 * shows that.
 */
class SessionHeartbeats {
  private mayBeOpen = true;
  private opened = false;

  constructor(private readonly ctx: Context) {
    // The Worker sends an attempt's first heartbeat at once.
    this.beat();
  }

  details(): string | undefined {
    if (this.opened) return SESSION_OPENED;
    if (this.mayBeOpen) return SESSION_MAY_BE_OPEN;
    return undefined;
  }

  beat(): void {
    try {
      this.ctx.heartbeat(this.details());
    } catch {
      // As in startAdaptiveHeartbeat: the Activity may already be cancelled.
    }
  }

  /**
   * Waits until the Temporal server holds this attempt's first heartbeat. Until
   * then, a lost attempt's retry would find no marker and open a new, empty
   * session.
   */
  async waitUntilStored(): Promise<void> {
    const deadline = Date.now() + STORE_TIMEOUT_MS;
    let lastError: unknown;
    const timedOut = () =>
      ApplicationFailure.create({
        message: `The server did not show the stateful MCP server session Activity's first heartbeat within ${STORE_TIMEOUT_MS / 1000} seconds.`,
        cause: lastError instanceof Error ? lastError : undefined,
      });
    let timer: ReturnType<typeof setTimeout> | undefined;
    // The deadline also bounds a describe that never returns, and the decoding
    // of the details.
    const expired = new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(timedOut()), STORE_TIMEOUT_MS);
    });
    const readBack = this.readBack(deadline, timedOut, (err) => {
      lastError = err;
    });
    // Nothing waits for it after the deadline.
    readBack.catch(() => {});
    try {
      await Promise.race([readBack, expired]);
    } finally {
      clearTimeout(timer);
    }
  }

  private async readBack(deadline: number, timedOut: () => Error, onError: (err: unknown) => void): Promise<void> {
    const { info, client } = this.ctx;
    const workflowId = info.workflowExecution!.workflowId;
    const handle = client.workflow.getHandle(workflowId, info.workflowExecution!.runId);
    const { payloadConverter, payloadCodecs } = client.options.loadedDataConverter;
    // The context the Worker encodes this Activity's heartbeats with.
    const context: SerializationContext = {
      type: 'activity',
      namespace: info.namespace,
      activityId: info.activityId,
      workflowId,
      isLocal: info.isLocal,
    };
    let delayMs = 100;
    while (Date.now() < deadline) {
      let description;
      try {
        description = await client.connection.withDeadline(deadline, () => handle.describe());
      } catch (err) {
        const refusal = describeRefusal(err);
        if (refusal) {
          const status = `${refusal.status}: ${refusal.details}`;
          const reason = DESCRIBE_PERMISSION_ERRORS.has(refusal.status)
            ? `the server refused it (${status}). Give the Worker's client permission to describe Workflows in this namespace.`
            : `the server could not describe this Workflow run (${status}).`;
          throw ApplicationFailure.create({
            message: `The stateful MCP server session Activity confirms its first heartbeat with DescribeWorkflowExecution, and ${reason}`,
            type: DEDICATED_WORKER_FAILURE_TYPE,
            nonRetryable: true,
            cause: err instanceof Error ? err : undefined,
          });
        }
        onError(err);
      }
      if (description) {
        const pending = (description.raw.pendingActivities ?? []).find((p) => p.activityId === info.activityId);
        if (!pending || pending.attempt !== info.attempt) {
          // A later attempt or the Workflow's end replaced this one, and the
          // details on the server are not this attempt's.
          throw ApplicationFailure.create({
            message: 'This stateful MCP server session Activity attempt is no longer the current one.',
          });
        }
        let details: unknown;
        try {
          let payloads = pending.heartbeatDetails?.payloads ?? [];
          for (const codec of [...payloadCodecs].reverse()) {
            payloads = await codec.decode(payloads, context);
          }
          details = payloads.length ? payloadConverter.fromPayload(payloads[0]!, context) : undefined;
        } catch (err) {
          throw ApplicationFailure.create({
            message: 'The stateful MCP server session Activity could not decode its own heartbeat details.',
            type: DEDICATED_WORKER_FAILURE_TYPE,
            nonRetryable: true,
            cause: err instanceof Error ? err : undefined,
          });
        }
        if (details === SESSION_MAY_BE_OPEN || details === SESSION_OPENED) return;
      }
      await this.ctx.sleep(delayMs);
      delayMs = Math.min(delayMs * 2, 2_000);
    }
    throw timedOut();
  }

  /** Records that the MCP server is connected. */
  recordOpened(): void {
    this.opened = true;
    this.beat();
  }

  /** Ends the attempt. The Worker sends the last details with its failure. */
  close(): void {
    if (this.mayBeOpen && !this.opened) {
      // This attempt connected no server, so a retry may connect again.
      this.mayBeOpen = false;
      this.beat();
    }
  }

  /**
   * The error that ends an attempt which threw `err`. Before a cancelled
   * result the Worker drops a heartbeat queued behind one still being encoded,
   * which can be the one that clears the marker. A Worker shutdown cancels the
   * attempt and is retried, so an attempt that connected no server fails
   * instead, and the Worker sends the clearing heartbeat first.
   */
  endingError(err: unknown): unknown {
    if (
      this.opened ||
      !this.ctx.cancellationDetails?.workerShutdown ||
      !(err instanceof CancelledFailure || (err instanceof Error && err.name === 'AbortError'))
    ) {
      return err;
    }
    return ApplicationFailure.create({
      message: 'The Worker shut down before this stateful MCP server session connected.',
      cause: err,
    });
  }
}

/**
 * Stateful MCP server provider. Maintains a persistent MCP connection per
 * Workflow run via a dedicated in-process Worker on a per-run Task Queue.
 *
 * Callers must handle `ApplicationFailure` with type `"DedicatedWorkerFailure"`
 * if the dedicated Worker fails to start or misses a heartbeat, or if the
 * session Activity is retried after it may have opened a session.
 */
export class StatefulMCPServerProvider {
  private readonly _name: string;
  private readonly _servers: Map<string, MCPServer> = new Map();
  private readonly _serverFactory: (factoryArgument?: unknown) => MCPServer;
  private readonly _nativeConnection: NativeConnection;

  /**
   * @param name - The internal name is `${name}-stateful` to avoid colliding with stateless providers using the same base name.
   * @param serverFactory - Must return a fresh server instance per call (one per Workflow run).
   * @param nativeConnection - NativeConnection used by the dedicated per-run Worker; typically the same one the main Worker uses.
   */
  constructor(
    name: string,
    serverFactory: (factoryArgument?: unknown) => MCPServer,
    nativeConnection: NativeConnection
  ) {
    this._name = `${name}${MCP_STATEFUL_SUFFIX}`;
    this._serverFactory = serverFactory;
    this._nativeConnection = nativeConnection;
  }

  get name(): string {
    return this._name;
  }

  _getActivities(): Record<string, (...args: any[]) => Promise<unknown>> {
    const serverId = (): string => {
      const info = activityInfo();
      // `workflowExecution` is always set for Activities scheduled from a Workflow,
      // which is the only way these per-Run Activities are dispatched.
      return `${this._name}@${info.workflowExecution!.runId}`;
    };

    const getServerOrThrow = (): MCPServer => {
      const id = serverId();
      const server = this._servers.get(id);
      if (!server) {
        throw ApplicationFailure.create({
          message: `No active server for ${id}`,
          type: 'StatefulMCPServerNotFound',
          nonRetryable: true,
        });
      }
      rejectCallableToolFilter(server);
      rejectToolMetaResolver(server);
      return server;
    };

    // Operation Activities — registered on the per-run Task Queue.
    const listTools = async () => {
      const server = getServerOrThrow();
      return filterListedTools(server, await server.listTools());
    };

    const callTool = async (input: CallToolArgs) =>
      callToolWithMeta(getServerOrThrow(), input.toolName, input.args, input.meta);

    // Long-running session Activity on the MAIN Worker's Task Queue. Creates
    // a server, connects it, spins up a dedicated per-run Worker, then awaits
    // until cancelled.
    const serverSession = async (input?: StatefulMcpServerSessionArgs): Promise<void> => {
      const ctx = Context.current();
      const info = ctx.info;
      const sid = `${this._name}@${info.workflowExecution!.runId}`;

      const failure = earlierAttemptFailure(info.heartbeatDetails);
      if (failure) {
        throw failure;
      }
      const heartbeats = new SessionHeartbeats(ctx);
      // Heartbeats so a slow `server.connect()` doesn't trip the timeout.
      const stopHeartbeat = startAdaptiveHeartbeat(() => heartbeats.details());

      try {
        // No server is built before the Temporal server holds the first
        // heartbeat, which says a session may be open.
        await heartbeats.waitUntilStored();
        // A cancellation, such as a Worker shutdown, doesn't interrupt the read-back.
        if (ctx.cancellationSignal.aborted) throw ctx.cancellationSignal.reason;
        if (this._servers.has(sid)) {
          throw ApplicationFailure.create({
            message:
              'Cannot connect to an already running server. Use a distinct name if running multiple servers in one workflow.',
            type: 'StatefulMCPServerAlreadyRunning',
            nonRetryable: true,
          });
        }

        const server = this._serverFactory(input?.factoryArgument);
        assertMCPServer(this.name, server);

        try {
          this._servers.set(sid, server);
          await server.connect();
          // Nor does every connect() observe it: close this server instead of serving.
          if (ctx.cancellationSignal.aborted) throw ctx.cancellationSignal.reason;
          heartbeats.recordOpened();

          const { Worker } = await import('@temporalio/worker');

          const dedicatedTaskQueue = sid;

          const dedicatedActivities: Record<string, (...args: any[]) => Promise<unknown>> = {
            [`${this._name}${MCP_LIST_TOOLS_SUFFIX}`]: listTools,
            [`${this._name}${MCP_CALL_TOOL_SUFFIX}`]: callTool,
          };

          // Wire the Activity interceptor identically to the main Worker so
          // tool Activities still restore trace context and emit
          // `temporal:executeActivity:*` spans. Options are forwarded per-run
          // via input; the tracing key resolver comes from the Worker running
          // this session, so it stays out of Activity arguments. Lazy import to
          // avoid pulling the Worker package unnecessarily.
          const { OpenAIAgentsTraceActivityInboundInterceptor, tracingApiKeyResolverFor } =
            await import('./trace-interceptor');
          const interceptorOpts = {
            ...input?.activityInterceptorOptions,
            tracingApiKey: tracingApiKeyResolverFor(Context.current()),
          };
          const dedicatedWorker = await Worker.create({
            connection: this._nativeConnection,
            taskQueue: dedicatedTaskQueue,
            activities: dedicatedActivities,
            maxConcurrentActivityTaskExecutions: 1,
            interceptors: {
              activity: [
                (ctx) => ({
                  inbound: new OpenAIAgentsTraceActivityInboundInterceptor(ctx, interceptorOpts),
                }),
              ],
            },
          });

          registerCancelShutdown(Context.current(), dedicatedWorker);

          await dedicatedWorker.run();
        } finally {
          await server.close();
          this._servers.delete(sid);
        }
      } catch (err) {
        throw heartbeats.endingError(err);
      } finally {
        stopHeartbeat();
        heartbeats.close();
      }
    };

    // Only the session Activity is registered on the main Worker; the
    // operation Activities live on the per-run Worker.
    return {
      [`${this._name}${MCP_SERVER_SESSION_SUFFIX}`]: serverSession,
    };
  }
}
