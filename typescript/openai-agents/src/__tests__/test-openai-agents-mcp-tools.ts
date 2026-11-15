import { createMCPToolStaticFilter } from '@openai/agents-core';
import type { MCPServer } from '@openai/agents-core';
import { WorkflowFailedError } from '@temporalio/client';
import type { ApplicationFailure } from '@temporalio/common';
import { helpers } from '@temporalio/test-helpers';
import type { NativeConnection } from '@temporalio/worker';
import {
  OpenAIAgentsPlugin,
  StatefulMCPServerProvider,
  StatelessMCPServerProvider,
  TOOL_FILTER_UNSUPPORTED_TYPE,
} from '..';
import {
  mcpFactoryArgWorkflow,
  mcpFactoryValidationWorkflow,
  mcpToolFilterResumeWorkflow,
  mcpToolFilterWorkflow,
  mcpToolMetaWorkflow,
  statefulMcpFactoryResolverWorkflow,
} from './workflows/openai-agents';
import { FakeModelProvider, textResponse, toolCallResponse, type FakeModel } from './stubs/openai-agents';
import { makeOpenAIAgentsTestFunction } from './helpers/openai-agents';

const test = makeOpenAIAgentsTestFunction();

function* mcpFactoryArgGenerator() {
  yield toolCallResponse('get_time', {});
  yield textResponse('The time for tenant-42 is 2026-01-01.');
}

test('factoryArgument is passed through to MCP activities', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  let receivedFactoryArg: unknown;
  const testMcp = new StatelessMCPServerProvider('testMcp', (factoryArgument?: unknown): MCPServer => {
    receivedFactoryArg = factoryArgument;
    return {
      cacheToolsList: false,
      name: 'testMcp',
      async connect() {},
      async close() {},
      async listTools() {
        return [
          {
            name: 'get_time',
            description: 'Returns current time',
            inputSchema: {
              type: 'object' as const,
              properties: {},
              required: [] as string[],
              additionalProperties: false,
            },
          },
        ];
      },
      async callTool(...callArgs: unknown[]) {
        t.deepEqual(factoryArgument, { tenantId: 'tenant-42' }, 'factoryArgument should be passed to callTool');
        t.is(callArgs.length, 2, 'callTool should get no meta argument when there is no toolMetaResolver');
        return [{ type: 'text', text: '2026-01-01' }];
      },
      async invalidateToolsCache() {},
    };
  });
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider(() => mcpFactoryArgGenerator()),
        mcpServerProviders: [testMcp],
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(mcpFactoryArgWorkflow, {
      args: ['What time is it?'],
      workflowExecutionTimeout: '30 seconds',
    });

    const result = await handle.result();
    t.truthy(result, 'Workflow should complete successfully');

    t.deepEqual(receivedFactoryArg, { tenantId: 'tenant-42' }, 'factoryArgument should be passed to listTools');
  });
});

function filterTestServer(name: string, toolFilter: MCPServer['toolFilter']): MCPServer {
  return {
    cacheToolsList: false,
    name,
    toolFilter,
    async connect() {},
    async close() {},
    async listTools() {
      return ['safe_tool', 'danger_tool', 'extra_tool'].map((toolName) => ({
        name: toolName,
        description: toolName,
        inputSchema: {
          type: 'object' as const,
          properties: {},
          required: [] as string[],
          additionalProperties: false,
        },
      }));
    },
    async callTool() {
      return [{ type: 'text', text: 'unused' }];
    },
    async invalidateToolsCache() {},
  };
}

function filterTestProviders(toolFilter: MCPServer['toolFilter'], nativeConnection: NativeConnection) {
  return [
    new StatelessMCPServerProvider('filterStateless', () => filterTestServer('filterStateless', toolFilter)),
    new StatefulMCPServerProvider('filterStateful', () => filterTestServer('filterStateful', toolFilter), nativeConnection),
  ];
}

test('MCP tool filters apply on stateless and stateful servers', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const modelProvider = new FakeModelProvider(Array.from({ length: 6 }, () => textResponse('done')));
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider,
        mcpServerProviders: filterTestProviders(
          createMCPToolStaticFilter({ blocked: ['danger_tool'] }),
          t.context.env.nativeConnection
        ),
      }),
    ],
  });

  await worker.runUntil(async () => {
    for (const [stateful, listToolsActivity] of [
      [false, 'filterStateless-stateless-list-tools'],
      [true, 'filterStateful-stateful-list-tools'],
    ] as const) {
      const handle = await startWorkflow(mcpToolFilterWorkflow, { args: [stateful], workflowExecutionTimeout: '30 seconds' });
      t.deepEqual(await handle.result(), ['done', 'done', 'done']);
      // The handle lists the server's tools once; the agent filters that list in every run.
      const { events } = await handle.fetchHistory();
      const listings = events?.filter(
        (e) => e.activityTaskScheduledEventAttributes?.activityType?.name === listToolsActivity
      );
      t.is(listings?.length, 1);
    }
  });

  // `danger_tool` is blocked by the factory server's static filter. The Workflow's filter allows
  // `extra_tool` only in the first run, whose context is admin.
  const offered = (modelProvider.getModel() as FakeModel).requests.map((r) => r.tools.map((tool) => tool.name));
  const perWorkflow = [['safe_tool', 'extra_tool'], ['safe_tool'], ['safe_tool']];
  t.deepEqual(offered, [...perWorkflow, ...perWorkflow]);
});

test('a callable toolFilter on the factory server fails the run', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([]),
        mcpServerProviders: filterTestProviders(async () => true, t.context.env.nativeConnection),
      }),
    ],
  });

  await worker.runUntil(async () => {
    for (const stateful of [false, true]) {
      const err = await t.throwsAsync(
        executeWorkflow(mcpToolFilterWorkflow, { args: [stateful], workflowExecutionTimeout: '30 seconds' }),
        { instanceOf: WorkflowFailedError }
      );
      t.is((err?.cause?.cause as ApplicationFailure | undefined)?.type, TOOL_FILTER_UNSUPPORTED_TYPE);
    }
  });
});

test('an invalid factory server fails stateless and stateful Activities without retries', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const factoryCalls = { stateless: 0, stateful: 0 };
  let connects = 0;
  let closes = 0;
  const invalidServer = {
    async connect() {
      connects++;
    },
    async close() {
      closes++;
    },
  } as MCPServer;
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([]),
        mcpServerProviders: [
          new StatelessMCPServerProvider('factoryStateless', () => {
            factoryCalls.stateless++;
            return invalidServer;
          }),
          new StatefulMCPServerProvider(
            'factoryStateful',
            () => {
              factoryCalls.stateful++;
              return invalidServer;
            },
            t.context.env.nativeConnection
          ),
        ],
      }),
    ],
  });
  await worker.runUntil(async () => {
    for (const activityName of ['factoryStateless-stateless-list-tools', 'factoryStateful-stateful-server-session']) {
      const handle = await startWorkflow(mcpFactoryValidationWorkflow, {
        args: [activityName],
        workflowExecutionTimeout: '15 seconds',
      });
      const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
      const failure = err?.cause?.cause as ApplicationFailure | undefined;
      t.is(failure?.type, 'MCPServerFactoryTypeError');
      t.true(failure?.nonRetryable);
    }
  });
  t.deepEqual(factoryCalls, { stateless: 1, stateful: 1 });
  t.is(connects, 0);
  t.is(closes, 0);
});

test('a tool filter that throws while resuming from a RunState fails the run', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([toolCallResponse('confirm', {}), textResponse('done')]),
        mcpServerProviders: filterTestProviders(undefined, t.context.env.nativeConnection),
      }),
    ],
  });

  await worker.runUntil(async () => {
    const err = await t.throwsAsync(
      executeWorkflow(mcpToolFilterResumeWorkflow, { workflowExecutionTimeout: '15 seconds' }),
      { instanceOf: WorkflowFailedError }
    );
    t.is((err?.cause as ApplicationFailure | undefined)?.type, 'AgentsWorkflowError');
  });
});

function metaTestServer(name: string, metas: unknown[]): MCPServer {
  return {
    cacheToolsList: false,
    name,
    async connect() {},
    async close() {},
    async listTools() {
      return [
        {
          name: 'echo_meta',
          description: 'Records the request _meta',
          inputSchema: {
            type: 'object' as const,
            properties: {},
            required: [] as string[],
            additionalProperties: false,
          },
        },
      ];
    },
    async callTool(_toolName: string, _args: Record<string, unknown> | null, meta?: Record<string, unknown> | null) {
      metas.push(meta);
      return [{ type: 'text', text: 'ok' }];
    },
    async invalidateToolsCache() {},
  };
}

test('MCP toolMetaResolver sends _meta on stateless and stateful servers', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  const metas: unknown[] = [];
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([
          toolCallResponse('echo_meta', {}),
          textResponse('stateless done'),
          toolCallResponse('echo_meta', {}),
          textResponse('stateful done'),
        ]),
        mcpServerProviders: [
          new StatelessMCPServerProvider('metaStateless', () => metaTestServer('metaStateless', metas)),
          new StatefulMCPServerProvider(
            'metaStateful',
            () => metaTestServer('metaStateful', metas),
            t.context.env.nativeConnection
          ),
        ],
      }),
    ],
  });

  await worker.runUntil(async () => {
    t.is(
      await executeWorkflow(mcpToolMetaWorkflow, { args: [false], workflowExecutionTimeout: '30 seconds' }),
      'stateless done'
    );
    t.is(
      await executeWorkflow(mcpToolMetaWorkflow, { args: [true], workflowExecutionTimeout: '30 seconds' }),
      'stateful done'
    );
  });

  t.deepEqual(metas, [
    { tenant: 'tenant-7', tool: 'echo_meta', server: 'metaStateless' },
    { tenant: 'tenant-7', tool: 'echo_meta', server: 'metaStateful' },
  ]);
});

test('Stateful MCP: a factory toolMetaResolver fails the operations', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);

  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([]),
        mcpServerProviders: [
          new StatefulMCPServerProvider(
            'factoryResolver',
            () => ({ ...metaTestServer('factoryResolver', []), toolMetaResolver: () => ({ tenant: 'tenant-7' }) }),
            t.context.env.nativeConnection
          ),
        ],
      }),
    ],
  });

  await worker.runUntil(async () => {
    const handle = await startWorkflow(statefulMcpFactoryResolverWorkflow, { workflowExecutionTimeout: '20 seconds' });
    t.deepEqual(await handle.result(), [
      'MCPToolMetaResolverUnsupported:true',
      'MCPToolMetaResolverUnsupported:true',
      'cleanup-success',
    ]);
  });
});
