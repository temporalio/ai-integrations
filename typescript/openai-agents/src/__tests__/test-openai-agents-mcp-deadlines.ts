import { ServiceError, WorkflowFailedError, WorkflowNotFoundError } from '@temporalio/client';
import { type PayloadCodec } from '@temporalio/common';
import { helpers } from '@temporalio/test-helpers';
import { OpenAIAgentsPlugin, StatefulMCPServerProvider } from '..';
import { memoryServer } from './helpers/stateful-mcp-host';
import { statefulMcpSessionEndedWorkflow } from './workflows/openai-agents';
import { FakeModelProvider } from './stubs/openai-agents';
import { makeOpenAIAgentsTestFunction, assertSessionRefused } from './helpers/openai-agents';

const test = makeOpenAIAgentsTestFunction();

for (const [kase, title] of [
  ['refused', 'a session Activity that may not describe its Workflow fails at once'],
  ['not found', 'a session Activity that cannot describe its Workflow run fails at once'],
  ['describe stalls', 'a session Activity whose describe never returns fails at the deadline'],
  ['decode stalls', 'a session Activity that never finishes decoding its read-back fails at the deadline'],
] as const) {
  test(`Stateful MCP: ${title}`, async (t) => {
    const { createWorker, startWorkflow } = helpers(t);
    let factoryCalls = 0;
    const describe = async (): Promise<never> => {
      if (kase === 'not found') {
        // What the client throws for a NOT_FOUND describe.
        throw new WorkflowNotFoundError('Workflow not found.', 'unknown', undefined);
      }
      if (kase === 'refused') {
        const grpc = Object.assign(new Error('7 PERMISSION_DENIED: Request refused.'), {
          code: 7,
          details: 'Request refused.',
          metadata: {},
        });
        throw new ServiceError('Failed to describe workflow', { cause: grpc });
      }
      return new Promise<never>(() => {});
    };
    // Never finishes decoding the details the session Activity reads back.
    const stallReadBack: PayloadCodec = {
      async encode(payloads) {
        return payloads;
      },
      async decode(payloads) {
        if (payloads.some((p) => Buffer.from(p.data ?? []).toString().includes('mcp-session-may-be-open'))) {
          await new Promise(() => {});
        }
        return payloads;
      },
    };
    const worker = await createWorker({
      plugins: [
        new OpenAIAgentsPlugin({
          modelProvider: new FakeModelProvider([]),
          mcpServerProviders: [
            new StatefulMCPServerProvider(
              'sessionEnded',
              () => {
                factoryCalls++;
                return memoryServer('sessionEnded', 'S');
              },
              t.context.env.nativeConnection
            ),
          ],
        }),
      ],
      ...(kase === 'decode stalls'
        ? { dataConverter: { payloadCodecs: [stallReadBack] } }
        : { interceptors: { client: { workflow: [{ describe }] } } }),
    });
    await worker.runUntil(async () => {
      const handle = await startWorkflow(statefulMcpSessionEndedWorkflow, {
        args: [false],
        workflowExecutionTimeout: '60 seconds',
      });
      if (kase !== 'refused' && kase !== 'not found') {
        // The attempt fails at the 30-second deadline and is retried.
        let failures: string[] = [];
        for (let i = 0; i < 400 && !failures.length; i++) {
          await new Promise((r) => setTimeout(r, 100));
          const { raw } = await handle.describe();
          failures = (raw.pendingActivities ?? [])
            .filter((p) => p.activityType?.name?.endsWith('-server-session') && p.lastFailure?.message)
            .map((p) => p.lastFailure!.message!);
        }
        await handle.terminate();
        t.deepEqual(failures, [
          "The server did not show the stateful MCP server session Activity's first heartbeat within 30 seconds.",
        ]);
        return;
      }
      const err = await t.throwsAsync(handle.result(), { instanceOf: WorkflowFailedError });
      assertSessionRefused(t, err);
      const messages: string[] = [];
      for (let cause: unknown = err; cause instanceof Error; cause = cause.cause) messages.push(cause.message);
      if (kase === 'not found') {
        t.true(
          messages.some((m) => m.includes('could not describe this Workflow run')),
          messages.join(' <- ')
        );
        t.false(
          messages.some((m) => m.includes('permission')),
          messages.join(' <- ')
        );
      } else {
        t.true(
          messages.some((m) => m.includes('permission to describe')),
          messages.join(' <- ')
        );
      }
    });
    // The session Activity failed before it built a server.
    t.is(factoryCalls, 0);
  });
}
