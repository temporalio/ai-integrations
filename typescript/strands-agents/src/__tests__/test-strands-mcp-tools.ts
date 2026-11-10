import type { JSONValue, McpClient, Message, ModelStreamEvent, StreamOptions, Tool } from '@strands-agents/sdk';
import type { TestFn } from 'ava';
import type { BaseContext } from '@temporalio/test-helpers';
import { test as anyTest, helpers } from '@temporalio/test-helpers';
import { Worker } from '@temporalio/worker';
import { StrandsPlugin } from '..';
import { setupTestEnv, StubMcpClient, StubModel, textTurn, toolCallTurn } from './helpers/strands';
import { mcpAgent, mcpPrefixedServersAgent, mcpServersAgent } from './workflows/strands';

const test = anyTest as TestFn<BaseContext>;

setupTestEnv(test);

test('TemporalMCPClient re-lists tools each turn by default and picks up mid-run additions', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const counters = { listCalls: 0 };
  const callToolNames: string[] = [];
  const schema = { type: 'object', properties: { path: { type: 'string' } } };
  const stubTool = (name: string): Tool =>
    ({ name, description: name, toolSpec: { name, description: name, inputSchema: schema } }) as unknown as Tool;

  // The server only exposes `readFile` after the first listing, so the agent
  // can dispatch it on the second turn only if the per-turn refresh ran.
  class GrowingMcpClient extends StubMcpClient {
    constructor() {
      super([{ name: 'listFiles', description: 'List files', inputSchema: schema }]);
    }
    override async listTools(): Promise<never> {
      counters.listCalls++;
      const tools = counters.listCalls === 1 ? [stubTool('listFiles')] : [stubTool('listFiles'), stubTool('readFile')];
      return tools as never;
    }
    override async callTool(tool: { name: string }, args: JSONValue): Promise<JSONValue> {
      callToolNames.push(tool.name);
      return super.callTool(tool, args);
    }
  }

  const worker = await createWorker({
    plugins: [
      new StrandsPlugin({
        models: {
          test: () =>
            new StubModel([
              toolCallTurn('listFiles', 'call_1', { path: '/' }),
              // `readFile` only exists after the mid-run refresh adds it.
              toolCallTurn('readFile', 'call_2', { path: '/a' }),
              textTurn('done'),
            ]),
        },
        mcpClients: { growingServer: () => new GrowingMcpClient() },
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(mcpAgent, { args: ['list files', 'growingServer'] });
    t.is(result, 'done');
  });

  // listTools ran more than once: the agent-init listing plus a refresh per tool round.
  t.true(counters.listCalls > 1, `expected multiple listTools calls, got ${counters.listCalls}`);
  // The agent dispatched `readFile`, proving the mid-run refresh registered it.
  t.deepEqual(callToolNames, ['listFiles', 'readFile']);
});

test('cacheTools: true lists tools once for the whole workflow', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const counters = { listCalls: 0 };
  class CountingListMcpClient extends StubMcpClient {
    override async listTools(): Promise<never> {
      counters.listCalls++;
      return super.listTools();
    }
  }
  const factory = (): McpClient =>
    new CountingListMcpClient([
      { name: 'listFiles', description: 'List files', inputSchema: { type: 'object', properties: {} } },
    ]);

  const worker = await createWorker({
    plugins: [
      new StrandsPlugin({
        models: {
          // Two tool-call turns: with caching off this would re-list each turn.
          test: () =>
            new StubModel([
              toolCallTurn('listFiles', 'call_1', { path: '/a' }),
              toolCallTurn('listFiles', 'call_2', { path: '/b' }),
              textTurn('done'),
            ]),
        },
        mcpClients: { cachedServer: factory },
      }),
    ],
  });

  await worker.runUntil(async () => {
    // Third arg enables cacheTools on the workflow's TemporalMCPClient.
    const result = await executeWorkflow(mcpAgent, { args: ['list files', 'cachedServer', true] });
    t.is(result, 'done');
  });

  t.is(counters.listCalls, 1);
});

test('a re-listed MCP tool name is refused only while another tool of the agent keeps it', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const schema = { type: 'object', properties: {} };
  const stubTool = (name: string): Tool =>
    ({ name, description: name, toolSpec: { name, description: name, inputSchema: schema } }) as unknown as Tool;
  const listings = new Map<string, number>();
  const calls: string[] = [];
  // Lists `first` at the agent's first listing and `later` from then on. A listing with `!fail` fails.
  class ChangingMcpClient extends StubMcpClient {
    constructor(
      private readonly label: string,
      private readonly first: string[],
      private readonly later: string[]
    ) {
      super([]);
    }
    override async listTools(): Promise<never> {
      const count = (listings.get(this.label) ?? 0) + 1;
      listings.set(this.label, count);
      const names = count === 1 ? this.first : this.later;
      if (names.includes('!fail')) throw new Error(`${this.label} is down`);
      return names.map(stubTool) as never;
    }
    override async callTool(tool: { name: string }, args: JSONValue): Promise<JSONValue> {
      calls.push(`${this.label}:${tool.name}`);
      return super.callTool(tool, args);
    }
  }

  // Runs one agent with a client per server, in order, each listing `[first, later]`. The model
  // calls each tool in `turns`, one per model call, then answers `done`. Each workflow task replays
  // the run, and its history replays at the end.
  const run = async (
    servers: Record<string, [string[], string[]]>,
    turns: string[],
    local = false,
    replace?: [string, number],
    prefix?: string
  ): Promise<{ result: string; calls: string[] }> => {
    listings.clear();
    calls.length = 0;
    const plugin = (): StrandsPlugin =>
      new StrandsPlugin({
        models: {
          test: () =>
            new StubModel([...turns.map((name, i) => toolCallTurn(name, `call_${i + 1}`, {})), textTurn('done')]),
        },
        mcpClients: Object.fromEntries(
          Object.entries(servers).map(([server, [first, later]]) => [
            server,
            () => new ChangingMcpClient(server, first, later),
          ])
        ),
      });
    const worker = await createWorker({ plugins: [plugin()], maxCachedWorkflows: 0 });
    return worker.runUntil(async () => {
      const handle = await startWorkflow(mcpServersAgent, {
        args: ['list files', Object.keys(servers), local, replace, prefix],
      });
      const result = await handle.result();
      await Worker.runReplayHistory(
        { workflowBundle: t.context.workflowBundle, plugins: [plugin()] },
        await handle.fetchHistory()
      );
      return { result, calls: [...calls] };
    });
  };

  const fixOne =
    "Set a prefix (or a different one) on one server's TemporalMCPClient, or drop the tool from one with its toolFilters.";
  const fixThis = (name: string) =>
    `Set a prefix (or a different one) on the server's TemporalMCPClient, or drop '${name}' with its toolFilters.`;
  // b starts listing a's tool after the first tool call, with the servers in either order.
  t.deepEqual(await run({ a: [['listFiles'], ['listFiles']], b: [['other'], ['listFiles']] }, ['listFiles']), {
    result:
      "refused: Tool with name 'listFiles' already registered. MCP servers 'a' and 'b' both list it. " +
      `${fixOne} Tools: listFiles, other.`,
    calls: ['a:listFiles'],
  });
  t.deepEqual(await run({ b: [['x'], ['x']], a: [['other'], ['x']] }, ['other']), {
    result: `refused: Tool with name 'x' already registered. MCP servers 'b' and 'a' both list it. ${fixOne} Tools: other, x.`,
    calls: ['a:other'],
  });
  // The same with prefix `gh` on both: the refusal also gives the servers' own name.
  const prefixed = { a: [['search'], ['search']], b: [['other'], ['other', 'search']] } as Record<string, [string[], string[]]>;
  t.deepEqual(await run(prefixed, ['gh_other'], false, undefined, 'gh'), {
    result:
      "refused: Tool with name 'gh_search' already registered. MCP servers 'a' (as 'search') and 'b' (as 'search') " +
      `both list it. ${fixOne} Tools: gh_other, gh_search.`,
    calls: ['b:other'],
  });
  // a and b start listing names that differ only by '-' or '_'.
  t.deepEqual(await run({ a: [['p'], ['x_y']], b: [['q'], ['x-y']] }, ['p']), {
    result:
      "refused: Tool name 'x-y' already exists as 'x_y'. MCP servers 'a' and 'b' both list it. " +
      `${fixOne} Tools: p, q.`,
    calls: ['a:p'],
  });
  // c starts listing a name that a tool in `tools` has, or one that differs from it only by '-' or
  // '_', or lists a name twice, or two that differ only by '-' or '_', or one Strands rejects. The
  // agent keeps the listFiles that c's refused listing drops, and gets none of its new tools.
  for (const [later, refusal] of [
    [
      ['echo'],
      "Tool with name 'echo' already registered. MCP server 'c' lists it, and another tool of the agent has it. " +
        fixThis('echo'),
    ],
    [
      ['take-note'],
      "Tool name 'take-note' already exists as 'take_note'. MCP server 'c' lists it, and another tool of the agent has " +
        `it. ${fixThis('take-note')}`,
    ],
    [['other', 'other'], "Tool with name 'other' already registered. MCP server 'c' lists it twice."],
    [['x-y', 'x_y'], "Tool name 'x_y' already exists as 'x-y'. MCP server 'c' lists both names."],
    [
      ['newTool', 'bad.name'],
      "Tool name must contain only alphanumeric characters, hyphens, and underscores. MCP server 'c' lists 'bad.name'. " +
        "Drop it with the toolFilters of the server's TemporalMCPClient.",
    ],
  ] as [string[], string][]) {
    t.deepEqual(await run({ c: [['listFiles'], later] }, ['listFiles'], true), {
      result: `refused: ${refusal} Tools: echo, listFiles, take_note.`,
      calls: ['c:listFiles'],
    });
  }
  // a's re-listing fails after b's, which adds a tool: the agent's tools stay as they were.
  const failed = await run({ a: [['p'], ['!fail']], b: [['other'], ['other', 'new']] }, ['p']);
  t.regex(failed.result, /^refused: .* Tools: other, p\.$/);
  t.deepEqual(failed.calls, ['a:p']);
  // x moves from b to a, with the servers in either order: the model calls it on a.
  for (const servers of [
    { b: [['x'], ['later']], a: [['other'], ['x']] },
    { a: [['other'], ['x']], b: [['x'], ['later']] },
  ] as Record<string, [string[], string[]]>[]) {
    t.deepEqual(await run(servers, ['other', 'x']), { result: 'done Tools: later, x.', calls: ['a:other', 'a:x'] });
  }
  // Workflow code puts a tool under the server's x in the first tool round: the server's tool is
  // back at the next model call.
  t.deepEqual(await run({ a: [['x'], ['x']] }, ['x', 'x'], false, ['x', 1]), {
    result: 'done Tools: x.',
    calls: ['a:x', 'a:x'],
  });
  // b drops r, which no other server lists: r is gone, and so is a tool put under it.
  for (const replace of [undefined, ['r', 1]] as ([string, number] | undefined)[]) {
    t.deepEqual(await run({ b: [['r'], ['s']] }, ['r'], false, replace), { result: 'done Tools: s.', calls: ['b:r'] });
  }
});

test('prefix and toolFilters on TemporalMCPClient rename and filter tools, and calls use the server names', async (t) => {
  const { createWorker, executeWorkflow, startWorkflow } = helpers(t);
  const schema = { type: 'object', properties: {} };
  const tools = ['echo', 'skip', 'other'].map((name) => ({ name, description: name, inputSchema: schema }));
  const calls: string[] = [];
  class RecordingMcpClient extends StubMcpClient {
    constructor(private readonly label: string) {
      super(tools);
    }
    override async callTool(tool: { name: string }, args: JSONValue): Promise<JSONValue> {
      calls.push(`${this.label}:${tool.name}`);
      return super.callTool(tool, args);
    }
  }
  // The tool names the model is offered at its first call.
  let offered: string[] | undefined;
  class OfferedModel extends StubModel {
    override async *stream(messages: Message[], options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
      offered ??= (options?.toolSpecs ?? []).map((spec) => spec.name).sort();
      yield* super.stream(messages, options);
    }
  }
  const plugin = (): StrandsPlugin =>
    new StrandsPlugin({
      models: {
        test: () =>
          new OfferedModel([toolCallTurn('a_echo', 'call_1', {}), toolCallTurn('b_other', 'call_2', {}), textTurn('done')]),
      },
      mcpClients: { a: () => new RecordingMcpClient('a'), b: () => new RecordingMcpClient('b') },
    });
  // Every workflow task replays the run, and its history replays at the end: the callback filter
  // runs in workflow code.
  const worker = await createWorker({ plugins: [plugin()], maxCachedWorkflows: 0 });
  await worker.runUntil(async () => {
    const handle = await startWorkflow(mcpPrefixedServersAgent, { args: ['go'] });
    t.is(await handle.result(), 'done Tools: a_echo, a_other, a_skip, b_other.');
    await Worker.runReplayHistory(
      { workflowBundle: t.context.workflowBundle, plugins: [plugin()] },
      await handle.fetchHistory()
    );
  });
  t.deepEqual(offered, ['a_echo', 'a_other', 'a_skip', 'b_other']);
  t.deepEqual(calls, ['a:echo', 'b:other']);

  // A prefix on the factory's McpClient fails the listing, with the fix in the error.
  const refusing = await createWorker({
    plugins: [
      new StrandsPlugin({
        models: { test: () => new StubModel([textTurn('done')]) },
        mcpClients: { p: () => new StubMcpClient(tools, { prefix: 'p' }) },
      }),
    ],
  });
  const refused = await refusing.runUntil(executeWorkflow(mcpServersAgent, { args: ['go', ['p']] }));
  t.true(
    refused.includes(
      "for MCP server 'p' sets prefix 'p'. Set it on the workflow-side TemporalMCPClient instead: " +
        "new TemporalMCPClient({ server: 'p', prefix: 'p' })."
    ),
    refused
  );
});
