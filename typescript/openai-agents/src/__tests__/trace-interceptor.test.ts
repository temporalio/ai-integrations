import test from 'ava';
import { getGlobalTraceProvider, withCustomSpan } from '@openai/agents-core';
import type { NexusStartOperationInput, NexusStartOperationOutput } from '@temporalio/worker';
import { injectAgentsTraceHeaderNexus } from '../common/headers';
import { OpenAIAgentsTraceNexusInboundInterceptor } from '../worker/trace-interceptor';
import type { TracingApiKeyContext } from '../worker/trace-interceptor';

test('Nexus handler spans get the key from tracingApiKey', async (t) => {
  getGlobalTraceProvider().setDisabled(false);
  const contexts: TracingApiKeyContext[] = [];
  const interceptor = new OpenAIAgentsTraceNexusInboundInterceptor({
    tracingApiKey: (context) => {
      contexts.push(context);
      return 'sk-resolved';
    },
  });
  const headers = injectAgentsTraceHeaderNexus(
    {},
    { traceName: 'caller-trace', traceId: 'trace_0123456789abcdef0123456789abcdef', spanId: 'span_0123456789abcdef' }
  );

  let spanKey: string | undefined;
  await interceptor.startOperation({ ctx: { headers } } as unknown as NexusStartOperationInput, async () => {
    await withCustomSpan(
      async (span) => {
        spanKey = span.tracingApiKey;
      },
      { data: { name: 'handler-span' } }
    );
    return {} as NexusStartOperationOutput;
  });

  t.is(spanKey, 'sk-resolved');
  t.deepEqual(contexts, [{ traceId: 'trace_0123456789abcdef0123456789abcdef', traceName: 'caller-trace' }]);
});
