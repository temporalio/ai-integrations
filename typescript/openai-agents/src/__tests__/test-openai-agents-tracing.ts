import {
  addTraceProcessor,
  getCurrentTrace,
  getGlobalTraceProvider,
  withCustomSpan,
  withTrace,
} from '@openai/agents-core';
import type { MCPServer, Model, ModelProvider, Span, SpanData, TracingProcessor } from '@openai/agents-core';
import { WithStartWorkflowOperation, WorkflowClient, WorkflowUpdateStage } from '@temporalio/client';
import { temporal } from '@temporalio/proto';
import { helpers } from '@temporalio/test-helpers';
import {
  OpenAIAgentsPlugin,
  OpenAIAgentsTraceClientInterceptor,
  StatefulMCPServerProvider,
  type TracingApiKeyContext,
} from '..';
import {
  statefulMcpIsolationWorkflow,
  tracingApiKeyWorkflow,
  tracingKeyRefusedWorkflow,
} from './workflows/openai-agents';
import { textResponse } from './stubs/openai-agents';
import { makeOpenAIAgentsTestFunction } from './helpers/openai-agents';
import EventType = temporal.api.enums.v1.EventType;

const test = makeOpenAIAgentsTestFunction();

// A model whose call opens a span in the model Activity, as real models do with their generation span.
class SpanningModelProvider implements ModelProvider {
  getModel(): Model {
    return {
      getResponse: () => withCustomSpan(async () => textResponse('done'), { data: { name: 'model-call' } }),
      getStreamedResponse: () => {
        throw new Error('not used');
      },
    };
  }
}

test("tracingApiKey gives spans that cross a Temporal boundary the key the Worker resolves, and a trace's own key is refused", async (t) => {
  const { createWorker, taskQueue } = helpers(t);

  getGlobalTraceProvider().setDisabled(false);
  const ended: Array<{ traceId: string; name: string; key: string | undefined }> = [];
  const recorder: TracingProcessor = {
    onTraceStart: async () => {},
    onTraceEnd: async () => {},
    onSpanStart: async () => {},
    onSpanEnd: async (span: Span<SpanData>) => {
      const data = span.spanData;
      ended.push({
        traceId: span.traceId,
        name: data.type === 'custom' ? data.name : data.type,
        key: span.tracingApiKey,
      });
    },
    forceFlush: async () => {},
    shutdown: async () => {},
  };
  addTraceProcessor(recorder);

  const runWorkflowId = `tracing-key-run-${Date.now()}`;
  const traceWorkflowId = `tracing-key-trace-${Date.now()}`;
  const mcpWorkflowId = `tracing-key-mcp-${Date.now()}`;
  const contexts: TracingApiKeyContext[] = [];
  // connect() runs in the session Activity on the main Worker; listTools() and
  // callTool() run on the dedicated per-run Worker.
  const mcpProvider = new StatefulMCPServerProvider(
    'isolationTest',
    (): MCPServer => ({
      cacheToolsList: false,
      name: 'isolationTest',
      connect: () => withCustomSpan(async () => {}, { data: { name: 'mcp-connect' } }),
      async close() {},
      listTools: () =>
        withCustomSpan(
          async () => [
            {
              name: 'get_marker',
              description: 'Returns marker',
              inputSchema: {
                type: 'object' as const,
                properties: {},
                required: [] as string[],
                additionalProperties: false,
              },
            },
          ],
          { data: { name: 'mcp-list-tools' } }
        ),
      callTool: () =>
        withCustomSpan(async () => [{ type: 'text', text: 'marker' }], { data: { name: 'mcp-call-tool' } }),
      async invalidateToolsCache() {},
    }),
    (t.context as any).env.nativeConnection
  );
  const worker = await createWorker({
    plugins: [
      new OpenAIAgentsPlugin({
        modelProvider: new SpanningModelProvider(),
        mcpServerProviders: [mcpProvider],
        tracingApiKey: (context) => {
          contexts.push(context);
          if (context.workflowId === runWorkflowId) return 'sk-resolved-run';
          if (context.workflowId === traceWorkflowId) return 'sk-resolved-trace';
          if (context.workflowId === mcpWorkflowId) return 'sk-resolved-mcp';
          return undefined;
        },
      }),
    ],
  });
  const wfClient = new WorkflowClient({
    connection: (t.context as any).env.connection,
    interceptors: [new OpenAIAgentsTraceClientInterceptor()],
  });

  let clientTraceId = '';
  const refusal = "The OpenAI Agents plugin cannot pass a trace's own tracing key";
  await worker.runUntil(async () => {
    // The run starts its own trace.
    await wfClient.execute(tracingApiKeyWorkflow, {
      taskQueue,
      workflowId: runWorkflowId,
      workflowExecutionTimeout: '30 seconds',
    });
    // The client starts the Workflows inside its trace.
    await withTrace('tracing-key-client-trace', async () => {
      clientTraceId = getCurrentTrace()!.traceId;
      await wfClient.execute(tracingApiKeyWorkflow, {
        taskQueue,
        workflowId: traceWorkflowId,
        workflowExecutionTimeout: '30 seconds',
      });
      await wfClient.execute(statefulMcpIsolationWorkflow, {
        taskQueue,
        workflowId: mcpWorkflowId,
        workflowExecutionTimeout: '30 seconds',
      });
    });

    // A trace's own key is refused in Workflow code before the model call, and at any other Temporal call.
    for (const entryPoint of [
      'run',
      'stream',
      'runConfig',
      'withTrace',
      'withTraceOnly',
      'traceContext',
      'activity',
      'asToolRunOptions',
      'asToolRunConfig',
    ] as const) {
      const handle = await wfClient.start(tracingKeyRefusedWorkflow, {
        taskQueue,
        workflowId: `tracing-key-refused-${entryPoint}-${Date.now()}`,
        args: [entryPoint],
        workflowExecutionTimeout: '30 seconds',
      });
      const result = await handle.result();
      t.true(result.startsWith(`refused: ${refusal}`), `${entryPoint}: ${result}`);
      const { events } = await handle.fetchHistory();
      const scheduled = events?.filter((e) => e.eventType === EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED) ?? [];
      t.is(scheduled.length, 0, `${entryPoint}: no Activity`);
    }
    // Uncaught, the refusal fails the Workflow Task, which is retried until the code changes.
    const uncaught = await wfClient.start(tracingApiKeyWorkflow, {
      taskQueue,
      workflowId: `tracing-key-uncaught-${Date.now()}`,
      args: ['sk-run'],
      workflowExecutionTimeout: '30 seconds',
    });
    let taskFailure: string | null | undefined;
    for (let i = 0; i < 50 && taskFailure == null; i++) {
      const { events } = await uncaught.fetchHistory();
      taskFailure = events?.find((e) => e.eventType === EventType.EVENT_TYPE_WORKFLOW_TASK_FAILED)
        ?.workflowTaskFailedEventAttributes?.failure?.message;
      if (taskFailure == null) await new Promise((resolve) => setTimeout(resolve, 100));
    }
    t.true(taskFailure?.startsWith(refusal), `an uncaught refusal fails the Workflow Task: ${taskFailure}`);
    await uncaught.terminate();
    // The client is refused when it calls a Workflow inside a trace with its own key.
    const keyedOptions = {
      taskQueue,
      workflowId: `tracing-key-client-key-${Date.now()}`,
      workflowExecutionTimeout: '30 seconds' as const,
    };
    const keyedHandle = wfClient.getHandle(keyedOptions.workflowId);
    const clientCalls = {
      start: () => wfClient.start(tracingApiKeyWorkflow, keyedOptions),
      signal: () => keyedHandle.signal('unused'),
      query: () => keyedHandle.query('unused'),
      startUpdate: () => keyedHandle.startUpdate('unused', { waitForStage: WorkflowUpdateStage.ACCEPTED }),
      signalWithStart: () => wfClient.signalWithStart(tracingApiKeyWorkflow, { ...keyedOptions, signal: 'unused' }),
      startUpdateWithStart: () =>
        wfClient.startUpdateWithStart('unused', {
          waitForStage: WorkflowUpdateStage.ACCEPTED,
          startWorkflowOperation: new WithStartWorkflowOperation(tracingApiKeyWorkflow, {
            ...keyedOptions,
            workflowIdConflictPolicy: 'FAIL',
          }),
        }),
    };
    await withTrace(
      'tracing-key-keyed-client-trace',
      async () => {
        for (const [name, call] of Object.entries(clientCalls)) {
          await t.throwsAsync(call, { message: (message) => message.startsWith(refusal) }, name);
        }
      },
      { tracingApiKey: 'sk-client' }
    );
  });

  const runTraceId = contexts.find((c) => c.workflowId === runWorkflowId && c.traceName !== undefined)?.traceId;
  t.truthy(runTraceId, 'the resolver should be asked about the run trace in the model Activity');
  const keysIn = (traceId: string | undefined, name: string) =>
    ended.filter((s) => s.traceId === traceId && s.name === name).map((s) => s.key);
  // Workflow spans reach the Worker through a sink after each Workflow Task, so wait for them.
  for (let i = 0; i < 50 && keysIn(clientTraceId, 'agent').length === 0; i++) {
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  t.deepEqual(keysIn(runTraceId, 'model-call'), ['sk-resolved-run'], 'Activity span of a run');
  t.deepEqual(keysIn(runTraceId, 'agent'), ['sk-resolved-run'], 'Workflow span of a run');
  t.deepEqual(keysIn(clientTraceId, 'model-call'), ['sk-resolved-trace'], 'Activity span in a client trace');
  t.deepEqual(keysIn(clientTraceId, 'agent'), ['sk-resolved-trace'], 'Workflow span in a client trace');
  t.deepEqual(keysIn(clientTraceId, 'mcp-connect'), ['sk-resolved-mcp'], 'stateful MCP span on the main Worker');
  t.deepEqual(keysIn(clientTraceId, 'mcp-list-tools'), ['sk-resolved-mcp'], 'listTools span on the dedicated Worker');
  t.deepEqual(keysIn(clientTraceId, 'mcp-call-tool'), ['sk-resolved-mcp'], 'callTool span on the dedicated Worker');
  t.like(
    contexts.find((c) => c.traceId === runTraceId && c.traceName !== undefined),
    { workflowId: runWorkflowId, workflowType: 'tracingApiKeyWorkflow' },
    'the Activity context names the Workflow that scheduled it'
  );
});
