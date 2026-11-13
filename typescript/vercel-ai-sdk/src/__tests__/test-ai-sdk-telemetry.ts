import { randomUUID } from 'crypto';
import { openai } from '@ai-sdk/openai';
import { context, propagation, trace } from '@opentelemetry/api';
import { AsyncLocalStorageContextManager } from '@opentelemetry/context-async-hooks';
import { ExportResultCode, W3CTraceContextPropagator } from '@opentelemetry/core';
import { Resource } from '@opentelemetry/resources';
import {
  BasicTracerProvider,
  SimpleSpanProcessor,
  type ReadableSpan,
  type SpanExporter,
} from '@opentelemetry/sdk-trace-base';
import { SEMRESATTRS_SERVICE_NAME } from '@opentelemetry/semantic-conventions';
import { WorkflowClient } from '@temporalio/client';
import type { OpenTelemetrySinks } from '@temporalio/interceptors-opentelemetry';
import {
  makeWorkflowExporter,
  OpenTelemetryActivityInboundInterceptor,
  OpenTelemetryActivityOutboundInterceptor,
  OpenTelemetryWorkflowClientCallsInterceptor,
  OpenTelemetryWorkflowClientInterceptor,
} from '@temporalio/interceptors-opentelemetry';
import { createBaseBundlerOptions, Worker } from '@temporalio/test-helpers';
import type { InjectedSinks } from '@temporalio/worker';
import { AiSdkPlugin } from '..';
import { makeAiSdkTestFunction, remoteTests } from './helpers/ai-sdk';
import { helloWorkflowGenerator, TestProvider } from './stubs/ai-sdk';
import { telemetryWorkflow } from './workflows/ai-sdk';

const test = makeAiSdkTestFunction();

test.serial('Telemetry', async (t) => {
  if (remoteTests) {
    t.timeout(120 * 1000);
  }
  let provider: BasicTracerProvider | undefined;
  try {
    const spans = Array<ReadableSpan>();

    const staticResource = new Resource({
      [SEMRESATTRS_SERVICE_NAME]: 'ts-test-otel-worker',
    });
    const traceExporter: SpanExporter = {
      export(spans_, resultCallback) {
        spans.push(...spans_);
        resultCallback({ code: ExportResultCode.SUCCESS });
      },
      async shutdown() {
        // Nothing to shutdown
      },
    };
    provider = new BasicTracerProvider({
      resource: staticResource,
      spanProcessors: [new SimpleSpanProcessor(traceExporter)],
    });
    provider.register({
      contextManager: new AsyncLocalStorageContextManager().enable(),
      propagator: new W3CTraceContextPropagator(),
    });
    const sinks: InjectedSinks<OpenTelemetrySinks> = {
      exporter: makeWorkflowExporter(traceExporter, staticResource), // eslint-disable-line @typescript-eslint/no-deprecated
    };

    const worker = await Worker.create({
      connection: t.context.env.nativeConnection,
      namespace: t.context.env.namespace,
      plugins: [
        new AiSdkPlugin({
          modelProvider: remoteTests ? openai : new TestProvider(helloWorkflowGenerator()),
        }),
      ],
      taskQueue: 'test-ai-telemetry',
      workflowsPath: require.resolve('./workflows/ai-sdk'),
      bundlerOptions: createBaseBundlerOptions(),

      interceptors: {
        client: {
          workflow: [new OpenTelemetryWorkflowClientCallsInterceptor()],
        },
        workflowModules: [require.resolve('./workflows/otel-interceptors')],
        activity: [
          (ctx) => ({
            inbound: new OpenTelemetryActivityInboundInterceptor(ctx),
            outbound: new OpenTelemetryActivityOutboundInterceptor(ctx),
          }),
        ],
      },
      sinks,
    });

    const client = new WorkflowClient({
      connection: t.context.env.connection,
      namespace: t.context.env.namespace,
      interceptors: [new OpenTelemetryWorkflowClientInterceptor()],
    });
    await worker.runUntil(async () => {
      await client.execute(telemetryWorkflow, {
        taskQueue: 'test-ai-telemetry',
        workflowId: randomUUID(),
        args: ['Tell me about recursion'],
      });
    });
    t.log(
      'exported spans:',
      spans.map(({ name }) => name)
    );
    // @ai-sdk/otel's OpenTelemetry integration names spans `${operationName} ${modelId}`
    // following the GenAI semantic conventions; generateText maps to `invoke_agent`.
    const generateSpan = spans.find(({ name }) => name === `invoke_agent gpt-4o-mini`);
    t.true(generateSpan !== undefined);
  } finally {
    await provider?.shutdown();
    context.disable();
    propagation.disable();
    trace.disable();
  }
});
