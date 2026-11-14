// Integration tests that need a real Worker. Pure-function tests live in
// the sibling `*.test.ts` files.
import { spawn } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import type { MCPServer } from '@openai/agents-core';
import { Context } from '@temporalio/activity';
import { WorkflowFailedError } from '@temporalio/client';
import { defaultPayloadConverter, type PayloadCodec } from '@temporalio/common';
import { helpers } from '@temporalio/test-helpers';
import { OpenAIAgentsPlugin, StatefulMCPServerProvider } from '..';
import { memoryServer } from './helpers/stateful-mcp-host';
import {
  statefulMcpIsolationWorkflow,
  statefulMcpSlowConnectHeartbeatWorkflow,
  statefulMcpRetriedWorkflow,
} from './workflows/openai-agents';
import { FakeModelProvider, textResponse } from './stubs/openai-agents';
import { makeOpenAIAgentsTestFunction, assertSessionRefused } from './helpers/openai-agents';

const test = makeOpenAIAgentsTestFunction();

test('Stateful MCP: a retried session Activity fails instead of opening a new session', async (t) => {
  const { createWorker, startWorkflow, taskQueue } = helpers(t);
  // Worker A runs in a child process, so the test can kill the session's host.
  const bundleDir = fs.mkdtempSync(path.join(os.tmpdir(), 'stateful-mcp-host-'));
  const codePath = path.join(bundleDir, 'bundle.js');
  fs.writeFileSync(codePath, t.context.workflowBundle.code);
  const host = spawn(
    process.execPath,
    [require.resolve('./helpers/stateful-mcp-host'), t.context.env.address, taskQueue, codePath],
    { stdio: ['ignore', 'pipe', 'inherit'] }
  );
  const lines: string[] = [];
  host.stdout.on('data', (d) => lines.push(...String(d).trim().split('\n')));
  const waitFor = async (line: string) => {
    for (let i = 0; i < 300 && !lines.includes(line); i++) await new Promise((r) => setTimeout(r, 100));
    t.true(lines.includes(line), `host printed ${line}`);
  };
  try {
    await waitFor('ready');
    const handle = await startWorkflow(statefulMcpRetriedWorkflow, { workflowExecutionTimeout: '90 seconds' });
    await waitFor('put');
    // Kill the host once the server holds "opened" and a periodic heartbeat has
    // followed it, so the refusal relies on periodic heartbeats keeping it.
    const details = async () => {
      const { raw } = await handle.describe();
      const session = (raw.pendingActivities ?? []).find((p) =>
        p.activityType?.name?.endsWith('-server-session')
      );
      const payload = session?.heartbeatDetails?.payloads?.[0];
      return payload ? defaultPayloadConverter.fromPayload(payload) : undefined;
    };
    for (let i = 0; i < 100 && (await details()) !== 'mcp-session-opened'; i++) {
      await new Promise((r) => setTimeout(r, 100));
    }
    await new Promise((r) => setTimeout(r, 3_500));
    t.is(await details(), 'mcp-session-opened');
    host.kill('SIGKILL');

    // Worker B takes the retried session Activity after the heartbeat timeout.
    let factoryCalls = 0;
    const worker = await createWorker({
      plugins: [
        new OpenAIAgentsPlugin({
          modelProvider: new FakeModelProvider([]),
          mcpServerProviders: [
            new StatefulMCPServerProvider(
              'retried',
              () => {
                factoryCalls++;
                return memoryServer('retried', 'B');
              },
              t.context.env.nativeConnection
            ),
          ],
        }),
      ],
    });
    await worker.runUntil(async () => {
      const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
      assertSessionRefused(t, err);
      const messages: string[] = [];
      for (let cause: unknown = err; cause instanceof Error; cause = cause.cause) messages.push(cause.message);
      t.true(
        messages.some((m) => m.includes('opened an MCP session')),
        messages.join(' <- ')
      );
    });
    // A new server would open a new MCP session without the old one's state.
    t.is(factoryCalls, 0);
  } finally {
    host.kill('SIGKILL');
    fs.rmSync(bundleDir, { recursive: true, force: true });
  }
});

test('Stateful MCP: a session Activity retried after a failed connect connects again', async (t) => {
  const { createWorker, executeWorkflow } = helpers(t);

  let factoryCalls = 0;
  const mcpProvider = new StatefulMCPServerProvider(
    'isolationTest',
    (): MCPServer => {
      const call = ++factoryCalls;
      const server = memoryServer('isolationTest', `server-${call}`);
      server.listTools = async () => [
        {
          name: 'get_marker',
          description: 'Returns marker',
          inputSchema: { type: 'object' as const, properties: {}, required: [] as string[], additionalProperties: false },
        },
      ];
      if (call === 1) {
        server.connect = async () => {
          throw new Error('first connect failed');
        };
      }
      return server;
    },
    t.context.env.nativeConnection
  );
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('unused')]),
        mcpServerProviders: [mcpProvider],
      }),
    ],
  });

  await worker.runUntil(async () => {
    const result = await executeWorkflow(statefulMcpIsolationWorkflow, { workflowExecutionTimeout: '30 seconds' });
    // No session was opened, so the retry connects again.
    t.regex(result, /session=server-2/);
    t.is(factoryCalls, 2);
  });
});

for (const ignoresCancel of [false, true]) {
  test(`Stateful MCP: a session Activity retried after a shutdown before connecting connects again${
    ignoresCancel ? ', with a connect() that ignores cancellation' : ''
  }`, async (t) => {
    const { createWorker, startWorkflow } = helpers(t);

    let connectCalls = 0;
    const plugin = () =>
      new OpenAIAgentsPlugin({
        modelProvider: new FakeModelProvider([textResponse('unused')]),
        mcpServerProviders: [
          new StatefulMCPServerProvider(
            'slowConnectTest',
            (): MCPServer => {
              const server = memoryServer('slowConnectTest', 'S');
              server.connect = async () => {
                // The first connect lasts until the Worker's shutdown cancels it,
                // or 3 seconds that the shutdown doesn't interrupt.
                if (++connectCalls === 1) {
                  await (ignoresCancel ? new Promise((r) => setTimeout(r, 3_000)) : Context.current().sleep(30_000));
                }
              };
              server.listTools = async () => [
                {
                  name: 'dummy',
                  description: 'Dummy tool',
                  inputSchema: { type: 'object' as const, properties: {}, required: [], additionalProperties: false },
                },
              ];
              return server;
            },
            t.context.env.nativeConnection
          ),
        ],
      });
    // Slows the encoding of "may be open" heartbeats. The first one reaches the
    // server before connect(); the clearing heartbeat then queues behind a
    // periodic one that is still being encoded when the shutdown cancels connect().
    let heartbeatDelays = 0;
    const slowHeartbeats: PayloadCodec = {
      async encode(payloads) {
        if (payloads.some((p) => Buffer.from(p.data ?? []).toString().includes('mcp-session-may-be-open'))) {
          heartbeatDelays++;
          await new Promise((r) => setTimeout(r, 2_000));
        }
        return payloads;
      },
      async decode(payloads) {
        return payloads;
      },
    };

    const first = await createWorker({ plugins: [plugin()], dataConverter: { payloadCodecs: [slowHeartbeats] } });
    const handle = await first.runUntil(async () => {
      const handle = await startWorkflow(statefulMcpSlowConnectHeartbeatWorkflow, { workflowExecutionTimeout: '40 seconds' });
      for (let i = 0; i < 300 && connectCalls === 0; i++) await new Promise((r) => setTimeout(r, 50));
      return handle;
    });
    t.is(connectCalls, 1);
    t.true(heartbeatDelays > 0);

    const second = await createWorker({ plugins: [plugin()] });
    await second.runUntil(async () => {
      // The first attempt connected no server, so the retry connects again.
      t.is(await handle.result(), 'connected:1');
    });
    t.is(connectCalls, 2);
  });
}
