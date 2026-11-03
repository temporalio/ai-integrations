import { McpClient } from '@strands-agents/sdk';
import type { JSONSchema, JSONValue } from '@strands-agents/sdk';
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';
import * as workflow from '@temporalio/workflow';
import type { ActivityOptions } from '@temporalio/workflow';
import { activityInfo, Context as ActivityContext } from '@temporalio/activity';

/** Tool descriptor returned by the per-server `{server}-listTools` activity. */
export interface McpToolInfo {
  name: string;
  description: string;
  inputSchema?: JSONSchema;
}

/** Activity input for the per-server `{server}-callTool` activity. */
export interface CallToolInput {
  toolName: string;
  args: JSONValue;
}

const STUB_TRANSPORT: Transport = {
  async start() {
    // No-op; TemporalMCPClient never connects from inside a workflow.
  },
  async send() {
    // No-op.
  },
  async close() {
    // No-op.
  },
};

/**
 * Options for {@link TemporalMCPClient}.
 *
 * The {@link ActivityOptions} apply to every per-tool activity invocation
 * the agent makes for this MCP server.
 *
 * `cacheTools` controls how often the agent re-lists this server's tools.
 * Strands lists an MCP client's tools once when the agent initializes and then
 * reuses that list for every turn, so the list is otherwise frozen for the
 * whole workflow.
 *
 * - `false` (default) — re-list the tools on each agent turn, so the agent
 *   picks up an MCP server that's restarted or redeployed mid-workflow. Costs
 *   one extra `{server}-listTools` activity per tool round.
 * - `true` — list once at the beginning of the workflow and reuse for all
 *   turns. Cheaper, but the tool list is fixed for the workflow's lifetime.
 */
export interface TemporalMCPClientOptions {
  server: string;
  activityOptions?: ActivityOptions;
  cacheTools?: boolean;
}

/**
 * Minimal view of Strands' `ToolRegistry` used by {@link
 * TemporalMCPClient.refreshTools}. Typed structurally because `ToolRegistry`
 * isn't re-exported from `@strands-agents/sdk`'s public index.
 */
export interface ToolRegistryView {
  list(): { name: string }[];
  remove(name: string): void;
  addOrReplace(tools: never[]): void;
}

/**
 * Workflow-side handle to an MCP server registered on the worker.
 *
 * The transport factory and tool discovery live worker-side via
 * `StrandsPlugin({ mcpClients: { server: () => new McpClient(...) } })`.
 * This handle only carries the server name (which selects the registered
 * factory) and the per-call activity options.
 *
 * Extends {@link McpClient} so it can be passed directly in an
 * {@link AgentConfig.tools | Agent.tools} array (Strands' `flattenTools`
 * dispatches on `instanceof McpClient`). `connect()` and `disconnect()`
 * are no-ops; `listTools()` returns lightweight {@link TemporalMCPTool}
 * wrappers whose `stream()` dispatches to the per-server `{server}-callTool`
 * activity registered by {@link StrandsPlugin}.
 */
export class TemporalMCPClient extends McpClient {
  private readonly server: string;
  private readonly activityOptions: ActivityOptions;
  /** Whether the agent lists this server's tools once or re-lists each turn. */
  readonly cacheTools: boolean;

  constructor(options: TemporalMCPClientOptions) {
    super({ transport: STUB_TRANSPORT });
    this.server = options.server;
    this.activityOptions = options.activityOptions ?? {};
    this.cacheTools = options.cacheTools ?? false;
  }

  override async connect(): Promise<void> {
    // No-op; all I/O happens inside the per-server callTool activity.
  }

  override async disconnect(): Promise<void> {
    // No-op.
  }

  // The base `McpClient.listTools` returns `Promise<McpTool[]>`, but `McpTool`
  // isn't re-exported from `@strands-agents/sdk`'s public index. The cast to
  // `never[]` keeps the override compatible; the agent only uses
  // `name`/`toolSpec`/`stream` on each tool, all of which
  // {@link TemporalMCPTool} provides.
  override async listTools(): Promise<never[]> {
    // Imported lazily to avoid a circular module cycle with TemporalMCPTool.
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { TemporalMCPTool } = require('./temporal-mcp-tool') as typeof import('./temporal-mcp-tool');
    const activities = workflow.proxyActivities<{
      [key: string]: (args: { server: string }) => Promise<McpToolInfo[]>;
    }>({
      startToCloseTimeout: '10 minutes',
      ...this.activityOptions,
    });
    const infos = await activities[listToolsActivityName(this.server)]!({ server: this.server });
    return infos.map((info) => new TemporalMCPTool(this.server, info, this.activityOptions)) as never[];
  }

  /**
   * Re-list this server's tools and reconcile them into the agent's tool
   * registry, dropping tools that no longer exist and adding new ones.
   * {@link TemporalAgent} calls this on each agent turn when `cacheTools` is
   * `false` so a mid-workflow MCP-server restart is reflected in the next call.
   */
  async refreshTools(registry: ToolRegistryView): Promise<void> {
    // Imported lazily to avoid a circular module cycle with TemporalMCPTool.
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { TemporalMCPTool } = require('./temporal-mcp-tool') as typeof import('./temporal-mcp-tool');
    const fresh = await this.listTools();
    const freshNames = new Set((fresh as unknown as { name: string }[]).map((t) => t.name));
    for (const tool of registry.list()) {
      if (tool instanceof TemporalMCPTool && tool.server === this.server && !freshNames.has(tool.name)) {
        registry.remove(tool.name);
      }
    }
    registry.addOrReplace(fresh);
  }
}

export function listToolsActivityName(server: string): string {
  return `${server}-listTools`;
}

export function callToolActivityName(server: string): string {
  return `${server}-callTool`;
}

/**
 * Builds the per-server `{server}-listTools` activity. Enumerates the server's
 * tools live over the plugin's connection on every call, so the agent sees the
 * server's current tools even if it is redeployed mid-workflow. Runs whenever a
 * workflow calls {@link TemporalMCPClient.listTools}.
 */
export function buildListToolsActivity(
  server: string,
  factory: () => McpClient,
  connections: McpConnections
): (input: { server: string }) => Promise<McpToolInfo[]> {
  const name = listToolsActivityName(server);
  const fn = async (_input: { server: string }): Promise<McpToolInfo[]> => {
    const record = connections.acquire(server, factory);
    try {
      const client = await record.client;
      const tools = await client.listTools();
      return tools.map((t) => ({
        name: t.name,
        description: t.description,
        inputSchema: t.toolSpec.inputSchema,
      }));
    } catch (err) {
      // The session may be broken (e.g. the server was redeployed); stop handing
      // it out so the next call reconnects to the current deployment. It is
      // closed once the calls still using it have finished.
      connections.retire(server, record);
      throw err;
    } finally {
      // Closes the connection if it was retired and this was the last in-flight
      // call; otherwise lets idle eviction resume.
      connections.release(server, record);
    }
  };
  Object.defineProperty(fn, 'name', { value: name });
  return fn;
}

/**
 * Default for how long an *idle* MCP connection is kept open before it's
 * disconnected. The window only starts once no {@link callToolActivityName |
 * callTool} or {@link listToolsActivityName | listTools} activity is using the
 * connection, so a call that runs longer than this is never cut off. Override
 * per plugin via `StrandsPlugin({ mcpConnectionIdleTimeout })`.
 */
export const MCP_CONNECTION_IDLE_MS = 5 * 60 * 1000;

/** A worker-process MCP connection plus the state governing its idle eviction. */
interface ConnectionRecord {
  /** Resolves to the connected client, or rejects if the handshake failed. */
  client: Promise<McpClient>;
  /** Number of activities currently using this connection. */
  inflight: number;
  /**
   * Set after a failed call, or when the last Worker using the plugin stops
   * while a call still uses the connection. A retired record is no longer
   * handed out to new calls, and is only disconnected once `inflight` drops
   * back to 0, so one call can't close the session under its siblings.
   */
  retired: boolean;
  /** Armed only while `inflight` is 0. */
  idleTimer?: NodeJS.Timeout;
}

function clearIdleTimer(record: ConnectionRecord): void {
  if (record.idleTimer !== undefined) {
    clearTimeout(record.idleTimer);
    record.idleTimer = undefined;
  }
}

/** Disconnect a record that is no longer handed out. Best-effort. */
async function closeRecord(record: ConnectionRecord): Promise<void> {
  try {
    const client = await record.client;
    await client.disconnect();
  } catch {
    // Best-effort; the session may already be broken.
  }
}

/**
 * The worker-process MCP connections of one {@link StrandsPlugin}, keyed by
 * server name. Workers that share a plugin instance share its connections;
 * another plugin instance opens its own through its own factories, even for
 * the same server name. The plugin's `runContext` counts the Workers using
 * them, and the last one to stop closes them. A connection that a call still
 * uses is closed when that call finishes.
 */
export class McpConnections {
  private readonly records: Map<string, ConnectionRecord> = new Map();
  private workers = 0;

  /**
   * @param idleMs how long an idle connection stays open; defaults to
   * {@link MCP_CONNECTION_IDLE_MS}.
   */
  constructor(private readonly idleMs: number = MCP_CONNECTION_IDLE_MS) {}

  /** Runs one Worker; once no Worker uses these connections, closes them. */
  async use(next: () => Promise<void>): Promise<void> {
    this.workers += 1;
    try {
      await next();
    } finally {
      this.workers -= 1;
      if (this.workers === 0) {
        await Promise.all([...this.records.keys()].map((server) => this.evict(server)));
      }
    }
  }

  /**
   * Stop handing out `server`'s connection, and disconnect it now or when the
   * last call still using it finishes.
   */
  async evict(server: string): Promise<void> {
    const record = this.records.get(server);
    if (record === undefined) return;
    // Delete before awaiting: a call arriving during `disconnect` must open a
    // fresh connection rather than join the one being torn down.
    this.records.delete(server);
    clearIdleTimer(record);
    if (record.inflight > 0) {
      record.retired = true;
      return;
    }
    await closeRecord(record);
  }

  /**
   * Return the cached connection for `server`, opening one lazily if needed,
   * and mark a call in flight so idle eviction can't disconnect underneath it.
   * Concurrent first-callers dedupe onto a single connect handshake by awaiting
   * the same record. Every caller must {@link release} once its call completes.
   */
  acquire(server: string, factory: () => McpClient): ConnectionRecord {
    let record = this.records.get(server);
    if (record === undefined) {
      const entry: ConnectionRecord = {
        inflight: 0,
        retired: false,
        client: (async () => {
          const client = factory();
          await client.connect();
          return client;
        })(),
      };
      // If the connect handshake fails, drop the rejected promise so the next
      // call retries instead of caching the failure forever.
      entry.client.catch(() => {
        if (this.records.get(server) === entry) this.records.delete(server);
      });
      this.records.set(server, entry);
      record = entry;
    }
    record.inflight += 1;
    clearIdleTimer(record);
    return record;
  }

  /**
   * Flag `record` as suspect after a failed call. It is unpublished at once so
   * new calls open a fresh connection, but not disconnected until the calls
   * still using it have released it (see {@link release}).
   */
  retire(server: string, record: ConnectionRecord): void {
    // Already evicted or replaced; whoever did that is closing it.
    if (this.records.get(server) !== record) return;
    record.retired = true;
    this.records.delete(server);
    clearIdleTimer(record);
  }

  /**
   * Mark a call done. Once no calls remain in flight, disconnect a retired
   * connection or arm idle eviction for a healthy one.
   */
  release(server: string, record: ConnectionRecord): void {
    record.inflight -= 1;
    if (record.retired) {
      if (record.inflight === 0) void closeRecord(record);
      return;
    }
    // Only the record still cached under this server arms a timer; a record
    // already evicted must not schedule one, or it could later evict a
    // different, healthy connection for the same server.
    if (record.inflight > 0 || this.records.get(server) !== record) return;
    const timer = setTimeout(() => {
      // `acquire` clears this timer synchronously, so a pending timer implies
      // the connection is still idle; re-checked to keep that invariant local.
      if (record.inflight === 0) void this.evict(server);
    }, this.idleMs);
    // Don't let an idle MCP connection keep the worker process alive.
    timer.unref?.();
    record.idleTimer = timer;
  }
}

/**
 * Builds the per-server `{server}-callTool` activity. Reuses the plugin's MCP
 * session, opened lazily through the user-supplied factory — successive calls
 * share one connection rather than reconnecting per call. Idle connections are
 * disconnected after the plugin's idle timeout.
 */
export function buildCallToolActivity(
  server: string,
  factory: () => McpClient,
  connections: McpConnections
): (input: CallToolInput) => Promise<JSONValue> {
  const name = callToolActivityName(server);
  const fn = async (input: CallToolInput): Promise<JSONValue> => {
    ActivityContext.current().log.debug(`Calling MCP tool ${input.toolName} on server ${server}`, {
      activityId: activityInfo().activityId,
    });
    const record = connections.acquire(server, factory);
    // Dispatch by name without re-listing tools on the connection: the agent
    // already has the schema from `listTools`, and `McpClient.callTool` only
    // reads `tool.name` to build the request. A minimal `{ name }` matches the
    // by-name dispatch the Python SDK does via `session.call_tool`.
    const tool = { name: input.toolName } as unknown as Parameters<McpClient['callTool']>[0];
    try {
      const client = await record.client;
      return await client.callTool(tool, input.args);
    } catch (err) {
      // The session may be broken; stop handing it out so the next call
      // reconnects. It is closed once the calls still using it have finished.
      connections.retire(server, record);
      throw err;
    } finally {
      // Closes the connection if it was retired and this was the last in-flight
      // call; otherwise lets idle eviction resume.
      connections.release(server, record);
    }
  };
  Object.defineProperty(fn, 'name', { value: name });
  return fn;
}
