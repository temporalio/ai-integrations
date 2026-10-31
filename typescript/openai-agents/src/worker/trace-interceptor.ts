import { withCustomSpan, getCurrentTrace } from '@openai/agents-core';
import type { Context as ActivityContext } from '@temporalio/activity';
import type {
  Next,
  ActivityInboundCallsInterceptor,
  ActivityExecuteInput,
  NexusInboundCallsInterceptor,
  NexusStartOperationInput,
  NexusStartOperationOutput,
} from '@temporalio/worker';
import { extractAgentsTraceHeader, extractAgentsTraceHeaderNexus } from '../common/headers';
import { withRestoredAgentsTraceContext } from '../common/trace-context';

/**
 * What the Worker knows about a trace it exports spans for. `traceName` is set
 * when it is known: in Activities, Nexus handlers and trace records.
 * `workflowId` and `workflowType` name the Workflow the span belongs to, or the
 * Workflow that scheduled the Activity; they are not set in Nexus handlers.
 */
export interface TracingApiKeyContext {
  traceId: string;
  traceName?: string;
  workflowId?: string;
  workflowType?: string;
}

/**
 * Returns the tracing export API key for a trace, Worker-side, or `undefined`
 * to leave the exporter's default key. The plugin calls it for spans that have
 * no key of their own: spans made in Activities (including stateful MCP
 * operations) and Nexus handlers, and Workflow spans in a trace started outside
 * the Workflow. A key that a run set in Workflow code with
 * `tracing: { apiKey }` applies to that run's Workflow spans only when the run
 * starts its own trace. Inside a trace started by a Client, the run joins that
 * trace and the Agents SDK ignores the run's key, so its Workflow spans get
 * the key this returns. The plugin does not send keys to the Workflow or write
 * them to Workflow history.
 *
 * Called often, so keep it fast. A throw fails the Activity or Nexus operation,
 * and drops the Workflow span with a warning.
 */
export type TracingApiKeyResolver = (context: TracingApiKeyContext) => string | undefined;

export interface OpenAIAgentsTraceInterceptorOptions {
  /**
   * Wrap intercepted calls in `temporal:*` custom spans for Temporal-specific
   * instrumentation. The plugin propagates this to Workflows via the
   * `__openai_agents_config` header. @default false
   */
  addTemporalSpans?: boolean;
  /** Worker-side tracing key for restored traces. See {@link TracingApiKeyResolver}. */
  tracingApiKey?: TracingApiKeyResolver;
}

// The resolver of the Worker that runs each Activity, so an Activity that
// starts its own Worker (stateful MCP) can pass it on without putting it in
// Activity arguments. The SDK gives interceptors the Activity's own Context.
const tracingApiKeyResolvers = new WeakMap<ActivityContext, TracingApiKeyResolver>();

/** The `tracingApiKey` resolver of the Worker running the Activity with this Context. */
export function tracingApiKeyResolverFor(ctx: ActivityContext): TracingApiKeyResolver | undefined {
  return tracingApiKeyResolvers.get(ctx);
}

/**
 * Activity inbound interceptor: restores agent trace context and optionally
 * wraps execution in a `temporal:executeActivity` custom span.
 */
export class OpenAIAgentsTraceActivityInboundInterceptor implements ActivityInboundCallsInterceptor {
  private readonly ctx: ActivityContext;

  constructor(
    ctx: ActivityContext,
    private readonly options?: OpenAIAgentsTraceInterceptorOptions
  ) {
    this.ctx = ctx;
    if (options?.tracingApiKey) tracingApiKeyResolvers.set(ctx, options.tracingApiKey);
  }

  async execute(input: ActivityExecuteInput, next: Next<ActivityInboundCallsInterceptor, 'execute'>): Promise<unknown> {
    const header = extractAgentsTraceHeader(input.headers);
    if (!header) return next(input);

    const addSpans = this.options?.addTemporalSpans === true;
    const info = this.ctx.info;
    const tracingApiKey = header.traceId
      ? this.options?.tracingApiKey?.({
          traceId: header.traceId,
          traceName: header.traceName,
          workflowId: info.workflowExecution?.workflowId,
          workflowType: info.workflowType,
        })
      : undefined;

    return withRestoredAgentsTraceContext(
      header,
      async () => {
        if (addSpans && getCurrentTrace()) {
          return withCustomSpan(() => next(input), {
            data: {
              name: `temporal:executeActivity:${info.activityType}`,
              data: {
                activityId: info.activityId,
                activityType: info.activityType,
              },
            },
          });
        }
        return next(input);
      },
      tracingApiKey
    );
  }
}

/**
 * Nexus inbound interceptor: restores agent trace context for `startOperation`
 * so handler-emitted spans nest under the Workflow-side trace tree.
 * `cancelOperation` is not wrapped.
 */
export class OpenAIAgentsTraceNexusInboundInterceptor implements NexusInboundCallsInterceptor {
  constructor(private readonly options?: OpenAIAgentsTraceInterceptorOptions) {}

  async startOperation(
    input: NexusStartOperationInput,
    next: Next<NexusInboundCallsInterceptor, 'startOperation'>
  ): Promise<NexusStartOperationOutput> {
    const header = extractAgentsTraceHeaderNexus(input.ctx.headers);
    if (!header?.traceId) return next(input);
    const tracingApiKey = this.options?.tracingApiKey?.({ traceId: header.traceId, traceName: header.traceName });
    return withRestoredAgentsTraceContext(header, () => next(input), tracingApiKey);
  }
}
