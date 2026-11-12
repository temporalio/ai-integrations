import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';
import type { BaseModelConfig, JSONValue, Message, ModelStreamEvent, StreamOptions, Tool } from '@strands-agents/sdk';
import { McpClient, Model } from '@strands-agents/sdk';
import type { TestFn } from 'ava';
import type { BaseContext } from '@temporalio/test-helpers';
import { createBaseBundlerOptions, TestWorkflowEnvironment } from '@temporalio/test-helpers';
import { bundleWorkflowCode, DefaultLogger } from '@temporalio/worker';
import { StrandsPlugin } from '../..';
import { startLocalWithRetry } from './environment';

const STUB_TRANSPORT: Transport = {
  async start() {},
  async send() {},
  async close() {},
};

/**
 * Stub `Model` whose `stream()` yields a deterministic sequence of events.
 * The Strands agent loop drives multiple model calls; this stub returns one
 * event sequence per call from the generator passed in.
 */
export class StubModel extends Model<BaseModelConfig> {
  private readonly turns: ModelStreamEvent[][];
  private turnIndex = 0;

  constructor(turns: ModelStreamEvent[][]) {
    super();
    this.turns = turns;
  }

  override updateConfig(_: BaseModelConfig): void {}
  override getConfig(): BaseModelConfig {
    return {};
  }

  override async *stream(_messages: Message[], _options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
    if (this.turnIndex >= this.turns.length) {
      throw new Error('StubModel exhausted: no more turns');
    }
    const events = this.turns[this.turnIndex++]!;
    for (const event of events) {
      yield event;
    }
  }
}

/** Convenience: text reply that ends the conversation. */
export function textTurn(text: string): ModelStreamEvent[] {
  return [
    { type: 'modelMessageStartEvent', role: 'assistant' },
    { type: 'modelContentBlockStartEvent' },
    { type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text } },
    { type: 'modelContentBlockStopEvent' },
    { type: 'modelMessageStopEvent', stopReason: 'endTurn' },
  ];
}

/** Convenience: tool-call turn that the agent loop will dispatch. */
export function toolCallTurn(toolName: string, toolUseId: string, input: object): ModelStreamEvent[] {
  return [
    { type: 'modelMessageStartEvent', role: 'assistant' },
    {
      type: 'modelContentBlockStartEvent',
      start: { type: 'toolUseStart', name: toolName, toolUseId },
    },
    {
      type: 'modelContentBlockDeltaEvent',
      delta: { type: 'toolUseInputDelta', input: JSON.stringify(input) },
    },
    { type: 'modelContentBlockStopEvent' },
    { type: 'modelMessageStopEvent', stopReason: 'toolUse' },
  ];
}

/**
 * Subclass of `McpClient` that returns a fixed tool list and a fixed call
 * result, with no real transport I/O. The per-server `{server}-listTools` and
 * `{server}-callTool` activities both call into this client.
 *
 * The plugin only reads `name`/`description`/`toolSpec.inputSchema` from the
 * returned tools and only uses `tool.name` when dispatching `callTool`, so the
 * stub returns minimal duck-typed objects.
 */
export class StubMcpClient extends McpClient {
  private readonly toolList: Tool[];

  constructor(
    toolList: Array<{ name: string; description: string; inputSchema: unknown }>,
    options: { prefix?: string } = {}
  ) {
    super({ transport: STUB_TRANSPORT, ...options });
    this.toolList = toolList.map(
      (t) =>
        ({
          name: t.name,
          description: t.description,
          toolSpec: {
            name: t.name,
            description: t.description,
            inputSchema: t.inputSchema,
          },
        }) as unknown as Tool
    );
  }

  override async connect(): Promise<void> {}
  override async disconnect(): Promise<void> {}
  override async listTools(): Promise<never> {
    // McpClient.listTools returns `Promise<McpTool[]>` and `McpTool` isn't in
    // the public index; widen via `never` for the stub.
    return this.toolList as never;
  }
  override async callTool(tool: { name: string }, args: JSONValue): Promise<JSONValue> {
    return {
      content: [{ type: 'text', text: `mcp:${tool.name}(${JSON.stringify(args)})` }],
      isError: false,
    } as never;
  }
}

/** Register the local server and workflow bundle for one integration test file. */
export function setupTestEnv(test: TestFn<BaseContext>): void {
  test.before(async (t) => {
    const env = await startLocalWithRetry(() => TestWorkflowEnvironment.createLocal());
    // Pass a `StrandsPlugin` to `bundleWorkflowCode` so the plugin can stub out
    // `@strands-agents/sdk/mcp-config.js`'s dynamic `node:*`/MCP-transport
    // imports and inline async chunks.
    const bundlerPlugin = new StrandsPlugin({ models: { test: () => new StubModel([]) } });
    const workflowBundle = await bundleWorkflowCode({
      ...createBaseBundlerOptions(),
      workflowsPath: require.resolve('../workflows/strands'),
      logger: new DefaultLogger('WARN'),
      plugins: [bundlerPlugin],
    });
    t.context = { env, workflowBundle };
  });

  test.after.always(async (t) => {
    await t.context.env?.teardown();
  });
}
