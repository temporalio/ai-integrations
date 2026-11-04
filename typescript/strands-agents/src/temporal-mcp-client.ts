import { McpClient, ToolValidationError } from '@strands-agents/sdk';
import type { JSONSchema, JSONValue } from '@strands-agents/sdk';
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';
import * as workflow from '@temporalio/workflow';
import type { ActivityOptions } from '@temporalio/workflow';
import { activityInfo, Context as ActivityContext } from '@temporalio/activity';
import type { TemporalMCPTool } from './temporal-mcp-tool';

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
 *
 * `prefix` and `toolFilters` work as on Strands' `McpClient`: the agent sees
 * `{prefix}_{name}`, and string and `RegExp` filters match the server's tool
 * name. A callback filter receives the workflow-side {@link TemporalMCPTool}
 * and runs in the workflow, so it must be deterministic.
 */
export interface TemporalMCPClientOptions {
  server: string;
  activityOptions?: ActivityOptions;
  cacheTools?: boolean;
  /** Prefix for the agent's tool names, applied as `{prefix}_{name}`. */
  prefix?: string;
  /** Which of the server's tools the agent gets. */
  toolFilters?: McpToolFilters;
}

/**
 * Matches a tool, as in Strands' `McpToolMatcher`: a string matches the
 * server's tool name exactly, a `RegExp` matches it from the start, and a
 * callback receives the tool under the name the agent sees.
 */
export type McpToolMatcher = string | RegExp | ((tool: TemporalMCPTool) => boolean);

/** Which of a server's tools the agent gets, as in Strands' `McpToolFilters`. */
export interface McpToolFilters {
  /** When present, only tools that match at least one matcher. */
  allowed?: McpToolMatcher[];
  /** Tools that match at least one matcher are left out, even when also allowed. */
  rejected?: McpToolMatcher[];
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
 * This handle carries the server name (which selects the registered
 * factory), the tool `prefix` and `toolFilters`, and the per-call activity
 * options.
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
  private readonly prefix: string | undefined;
  private readonly toolFilters: McpToolFilters | undefined;
  /** Whether the agent lists this server's tools once or re-lists each turn. */
  readonly cacheTools: boolean;
  // The names this client's tools hold in each agent's registry: the names of the tools Strands
  // added there, then the names of each refresh. While the server lists a name, the name is this
  // client's, whatever tool workflow code puts under it.
  private readonly held = new WeakMap<ToolRegistryView, Set<string>>();
  // The tools this client listed, to find its names in a registry it has no record for.
  private readonly listed = new WeakSet<object>();

  constructor(options: TemporalMCPClientOptions) {
    super({ transport: STUB_TRANSPORT });
    this.server = options.server;
    this.activityOptions = options.activityOptions ?? {};
    this.cacheTools = options.cacheTools ?? false;
    this.prefix = options.prefix;
    this.toolFilters = options.toolFilters;
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
    const tools = infos
      .map((info) => new TemporalMCPTool(this.server, info, this.activityOptions, this.prefix))
      .filter((tool) => included(tool, this.toolFilters));
    for (const tool of tools) this.listed.add(tool);
    return tools as never[];
  }

  /**
   * Re-list this server's tools and reconcile them into the agent's tool
   * registry, dropping tools that no longer exist and adding new ones, so a
   * mid-workflow MCP-server restart is reflected in the next call.
   * {@link TemporalAgent} instead re-lists each of its `cacheTools: false`
   * clients in its own hook after each tool round, and checks and applies
   * their listings together, so a name can move between its servers.
   *
   * A listed name that another tool of the agent has, that differs from one
   * only by `-` or `_`, that the listing has twice, or that Strands rejects
   * throws Strands' `ToolValidationError` and leaves the registry as it was.
   */
  async refreshTools(registry: ToolRegistryView): Promise<void> {
    const tools = (await this.listTools()) as unknown as TemporalMCPTool[];
    TemporalMCPClient.apply(registry, [{ client: this, tools }]);
  }

  /**
   * One `AfterToolsEvent` hook per client in `clients`, the `cacheTools: false` clients of one agent
   * in `tools` order, to register in that order. Strands runs them last to first. Each lists its
   * server as {@link refreshTools} does, so the `listTools` activities keep the order and timing
   * they had with one `refreshTools()` hook per client, unless a re-listing is refused: then every
   * server has listed before the refusal. The last client's hook starts the tool round, and the
   * first client's checks the round's listings together and applies them. So a name can move from
   * one server to another between tool rounds, and a name that two of the new listings share is
   * refused.
   *
   * @internal
   */
  static refreshHooks(clients: TemporalMCPClient[]): ((event: RefreshEvent) => Promise<void>)[] {
    const round: Listing[] = [];
    return clients.map(
      (client, i) => (event: RefreshEvent) =>
        client.listInRound(event.agent.toolRegistry, round, i === clients.length - 1, i === 0)
    );
  }

  // List this server into `round`, which `start` empties first, and apply the round at its `end`.
  private async listInRound(registry: ToolRegistryView, round: Listing[], start: boolean, end: boolean): Promise<void> {
    if (start) round.length = 0;
    const tools = (await this.listTools()) as unknown as TemporalMCPTool[];
    round.unshift({ client: this, tools });
    if (end) TemporalMCPClient.apply(registry, round);
  }

  /**
   * Make each of `clients` hold the name of each tool it listed that Strands adds to `registry`,
   * the agent's registry, when Strands adds it. Strands adds a listing tool by tool and stops at a
   * refused one, and a listing of an initialization that failed can add its tools after the agent
   * initialized again, so only the tools that are added count, whenever they are.
   *
   * @internal
   */
  static holdAdded(clients: TemporalMCPClient[], registry: ToolRegistryView & { add(tool: never): void }): void {
    const add = registry.add.bind(registry);
    registry.add = (tool: never) => {
      try {
        add(tool);
      } finally {
        const present = new Set<object>(registry.list());
        for (const added of [tool].flat() as { name: string }[]) {
          const client = clients.find((c) => c.listed.has(added));
          if (client === undefined || !present.has(added)) continue;
          const names = client.held.get(registry) ?? new Set();
          client.held.set(registry, names.add(added.name));
        }
      }
    };
  }

  // The names this client's tools hold in `registry`.
  private heldIn(registry: ToolRegistryView): Set<string> {
    return this.held.get(registry) ?? new Set(registry.list().flatMap((t) => (this.listed.has(t) ? [t.name] : [])));
  }

  // Check `listings` against each other, the agent's other tools and Strands' name rules, then apply
  // them: drop the names each client held and no longer lists, and add or replace the listed tools.
  // A refusal throws Strands' `ToolValidationError` and changes nothing.
  private static apply(registry: ToolRegistryView, listings: Listing[]): void {
    // Imported lazily to avoid a circular module cycle with TemporalMCPTool.
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { TemporalMCPTool } = require('./temporal-mcp-tool') as typeof import('./temporal-mcp-tool');
    const held = new Map(listings.map(({ client }) => [client, client.heldIn(registry)]));
    const freed = new Set([...held.values()].flatMap((names) => [...names]));
    const taken = new Map<string, { name: string }>();
    for (const tool of registry.list()) {
      if (!freed.has(tool.name)) taken.set(normalized(tool.name), tool);
    }
    const seen = new Map<string, TemporalMCPTool>();
    for (const { client, tools } of listings) {
      const own = new Map<string, TemporalMCPTool>();
      for (const tool of tools) {
        const { name } = tool;
        const rejected = rejection(name);
        if (rejected !== undefined) {
          throw new ToolValidationError(
            `${rejected}. MCP server ${listedAs(tool)} lists '${name}'. Drop it with the toolFilters of the ` +
              `server's TemporalMCPClient.`
          );
        }
        const first = own.get(normalized(name));
        if (first !== undefined) {
          const both = first.name === name ? 'it twice' : 'both names';
          throw duplicateToolName(name, first.name, `MCP server '${client.server}' lists ${both}.`);
        }
        own.set(normalized(name), tool);
        const earlier = seen.get(normalized(name));
        if (earlier !== undefined) throw duplicateToolName(name, earlier.name, clash(earlier, tool));
        const other = taken.get(normalized(name));
        if (other !== undefined) {
          const source =
            other instanceof TemporalMCPTool && !client.listed.has(other)
              ? clash(other, tool)
              : `MCP server ${listedAs(tool)} lists it, and another tool of the agent has it. Set a prefix (or a ` +
                `different one) on the server's TemporalMCPClient, or drop '${tool.serverToolName}' with its toolFilters.`;
          throw duplicateToolName(name, other.name, source);
        }
      }
      for (const [key, tool] of own) seen.set(key, tool);
    }
    for (const { client, tools } of listings) {
      const names = new Set(tools.map((t) => t.name));
      for (const name of held.get(client)!) {
        if (!names.has(name)) registry.remove(name);
      }
    }
    for (const { client, tools } of listings) {
      registry.addOrReplace(tools as never[]);
      client.held.set(registry, new Set(tools.map((t) => t.name)));
    }
  }
}

/** The part of Strands' `AfterToolsEvent` that {@link TemporalMCPClient.refreshHooks} reads. */
interface RefreshEvent {
  agent: { toolRegistry: ToolRegistryView };
}

// One client's new listing.
interface Listing {
  client: TemporalMCPClient;
  tools: TemporalMCPTool[];
}

// Strands' McpClient filter rule: whether `filters` let the agent have `tool`.
function included(tool: TemporalMCPTool, filters: McpToolFilters | undefined): boolean {
  if (!filters) return true;
  if (filters.allowed !== undefined && !filters.allowed.some((m) => matches(tool, m))) return false;
  return !(filters.rejected !== undefined && filters.rejected.some((m) => matches(tool, m)));
}

function matches(tool: TemporalMCPTool, matcher: McpToolMatcher): boolean {
  if (typeof matcher === 'function') return matcher(tool);
  if (typeof matcher === 'string') return matcher === tool.serverToolName;
  // The sticky flag anchors the match at the start of the name, as Strands does.
  const anchored = new RegExp(matcher.source, matcher.flags.includes('y') ? matcher.flags : `${matcher.flags}y`);
  return anchored.test(tool.serverToolName);
}

// Strands' words when its ToolRegistry rejects a tool name, or undefined when it accepts it.
function rejection(name: string): string | undefined {
  if (name.length < 1 || name.length > 64) return 'Tool name must be between 1 and 64 characters';
  if (!/^[a-zA-Z0-9_-]+$/.test(name)) return 'Tool name must contain only alphanumeric characters, hyphens, and underscores';
  return undefined;
}

// Strands' ToolRegistry refuses a name that differs from a registered one only by '-' or '_'.
function normalized(name: string): string {
  return name.replaceAll('-', '_');
}

// The server that lists `tool`, with the server's own name for it when a prefix renames it.
function listedAs(tool: TemporalMCPTool): string {
  return tool.serverToolName === tool.name ? `'${tool.server}'` : `'${tool.server}' (as '${tool.serverToolName}')`;
}

// Where two clashing MCP tools of two clients come from: `held` is the agent's, and `tool` the new one.
function clash(held: TemporalMCPTool, tool: TemporalMCPTool): string {
  return held.server === tool.server
    ? `The agent has two TemporalMCPClient handles for MCP server '${tool.server}'. Give one a different ` +
        `prefix, narrow one's toolFilters, or pass only one.`
    : `MCP servers ${listedAs(held)} and ${listedAs(tool)} both list it. Set a prefix (or a different one) on ` +
        `one server's TemporalMCPClient, or drop the tool from one with its toolFilters.`;
}

// Strands' words when its ToolRegistry refuses a name, then where the two tools come from.
function duplicateToolName(name: string, other: string, source: string): ToolValidationError {
  const refused =
    other === name ? `Tool with name '${name}' already registered.` : `Tool name '${name}' already exists as '${other}'.`;
  return new ToolValidationError(`${refused} ${source}`);
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
      refuseFactoryPrefix(server, client);
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

// A prefix on the factory's McpClient renames the tools the agent sees, and the callTool activity
// then asks the server for those names. Refuse it, retryably, so a run waits for a fixed worker.
function refuseFactoryPrefix(server: string, client: McpClient): void {
  // Strands' McpClient keeps the option in a private field.
  const prefix = (client as unknown as { _prefix?: string })._prefix;
  if (prefix) {
    throw new Error(
      `The McpClient that the StrandsPlugin mcpClients factory returns for MCP server '${server}' sets ` +
        `prefix '${prefix}'. Set it on the workflow-side TemporalMCPClient instead: ` +
        `new TemporalMCPClient({ server: '${server}', prefix: '${prefix}' }).`
    );
  }
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
