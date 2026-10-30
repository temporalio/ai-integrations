import test from 'ava';
import { getGlobalTraceProvider, withCustomSpan, withTrace } from '@openai/agents-core';
import type { NexusStartOperationInput, NexusStartOperationOutput } from '@temporalio/worker';
import { currentAgentsSpanHeader, injectAgentsTraceHeaderNexus } from '../common/headers';
import { OpenAIAgentsTraceNexusInboundInterceptor } from '../worker/trace-interceptor';
import type { TracingApiKeyContext } from '../worker/trace-interceptor';

test("Nexus handler spans get the key from tracingApiKey, and only a trace's own key is refused", async (t) => {
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
    // The handler can still call Temporal under the restored trace: the resolved key is not the trace's own.
    t.is(currentAgentsSpanHeader().traceId, 'trace_0123456789abcdef0123456789abcdef');
    await withTrace('handler-trace', async () => t.throws(() => currentAgentsSpanHeader()), {
      tracingApiKey: 'sk-own',
    });
    return {} as NexusStartOperationOutput;
  });

  t.is(spanKey, 'sk-resolved');
  t.deepEqual(contexts, [{ traceId: 'trace_0123456789abcdef0123456789abcdef', traceName: 'caller-trace' }]);
});
