import test from 'ava';
import { getCurrentTrace, getGlobalTraceProvider, withCustomSpan, withTrace } from '@openai/agents-core';
import type { Context as ActivityContext } from '@temporalio/activity';
import type { WorkflowStartInput, WorkflowStartOutput } from '@temporalio/client';
import type { ActivityExecuteInput, NexusStartOperationInput, NexusStartOperationOutput } from '@temporalio/worker';
import { OpenAIAgentsTraceClientInterceptor } from '../client/trace-interceptor';
import {
  currentAgentsSpanHeader,
  extractAgentsTraceHeader,
  injectAgentsTraceHeader,
  injectAgentsTraceHeaderNexus,
} from '../common/headers';
import { OpenAIAgentsTraceActivityInboundInterceptor, OpenAIAgentsTraceNexusInboundInterceptor } from '../worker/trace-interceptor';
import type { TracingApiKeyContext } from '../worker/trace-interceptor';

const callerTrace = {
  traceName: 'caller-trace',
  traceId: 'trace_0123456789abcdef0123456789abcdef',
  spanId: 'span_0123456789abcdef',
};

test("Nexus handler spans get the key from tracingApiKey, and only a trace's own key is refused", async (t) => {
  getGlobalTraceProvider().setDisabled(false);
  const contexts: TracingApiKeyContext[] = [];
  const interceptor = new OpenAIAgentsTraceNexusInboundInterceptor({
    tracingApiKey: (context) => {
      contexts.push(context);
      return 'sk-resolved';
    },
  });
  const headers = injectAgentsTraceHeaderNexus({}, callerTrace);

  let spanKey: string | undefined;
  await interceptor.startOperation({ ctx: { headers } } as unknown as NexusStartOperationInput, async () => {
    await withCustomSpan(
      async (span) => {
        spanKey = span.tracingApiKey;
        // Span helpers clone the restored trace; the clone keeps the resolved key's exemption.
        t.is(currentAgentsSpanHeader().traceId, callerTrace.traceId);
        await withCustomSpan(
          async () => {
            // A key that application code puts on the trace in place of the resolved one is its own.
            getCurrentTrace()!.tracingApiKey = 'sk-application';
            t.throws(() => currentAgentsSpanHeader(), { message: /cannot pass a trace's own tracing key/ });
          },
          { data: { name: 'replaced-key' } }
        );
        t.is(currentAgentsSpanHeader().traceId, callerTrace.traceId);
      },
      { data: { name: 'handler-span' } }
    );
    // The handler can still call Temporal under the restored trace: the resolved key is not the trace's own.
    t.is(currentAgentsSpanHeader().traceId, callerTrace.traceId);
    await withTrace('handler-trace', async () => t.throws(() => currentAgentsSpanHeader()), {
      tracingApiKey: 'sk-own',
    });
    return {} as NexusStartOperationOutput;
  });

  t.is(spanKey, 'sk-resolved');
  t.deepEqual(contexts, [{ traceId: callerTrace.traceId, traceName: callerTrace.traceName }]);
});

test('An Activity can start a Workflow under its restored trace with temporal spans on either side', async (t) => {
  getGlobalTraceProvider().setDisabled(false);
  const activityCtx = {
    info: {
      activityId: '1',
      activityType: 'startChild',
      workflowType: 'Parent',
      workflowExecution: { workflowId: 'parent' },
    },
  } as unknown as ActivityContext;
  const startInput = {
    workflowType: 'Child',
    options: { workflowId: 'child' },
    headers: {},
  } as unknown as WorkflowStartInput;

  const startFromActivity = (activitySpans: boolean, clientSpans: boolean, replaceKey: boolean) => {
    const activity = new OpenAIAgentsTraceActivityInboundInterceptor(activityCtx, {
      addTemporalSpans: activitySpans,
      tracingApiKey: () => 'sk-resolved',
    });
    const client = new OpenAIAgentsTraceClientInterceptor({ addTemporalSpans: clientSpans });
    const sent: WorkflowStartInput[] = [];
    const executed = activity.execute(
      { headers: injectAgentsTraceHeader({}, callerTrace) } as unknown as ActivityExecuteInput,
      async () => {
        if (replaceKey) getCurrentTrace()!.tracingApiKey = 'sk-application';
        return client.startWithDetails(startInput, async (input) => {
          sent.push(input);
          return {} as WorkflowStartOutput;
        });
      }
    );
    return { executed, sent };
  };

  for (const [activitySpans, clientSpans] of [
    [false, false],
    [true, false],
    [false, true],
    [true, true],
  ] as const) {
    const label = `activity spans ${activitySpans}, client spans ${clientSpans}`;
    const { executed, sent } = startFromActivity(activitySpans, clientSpans, false);
    await executed;
    t.is(sent.length, 1, label);
    const header = extractAgentsTraceHeader(sent[0]!.headers)!;
    t.is(header.traceId, callerTrace.traceId, label);
    t.false(JSON.stringify(header).includes('sk-'), label);

    const replaced = startFromActivity(activitySpans, clientSpans, true);
    await t.throwsAsync(replaced.executed, { message: /cannot pass a trace's own tracing key/ }, label);
    t.is(replaced.sent.length, 0, label);
  }
});
