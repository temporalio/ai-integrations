import * as otel from '@opentelemetry/api';
import { getCurrentTraceContext } from '@openai/agents-core';
import { defaultPayloadConverter, type Headers } from '@temporalio/common';
import type { SerializableModelActivityOptions } from './model-activity-options';

export const AGENTS_TRACE_HEADER_KEY = '__openai_span';

export interface AgentsSpanHeader {
  traceName: string;
  spanId: string | null;
  traceId: string | null;
  otelSpanId?: string;
}

export function injectAgentsTraceHeader(headers: Headers, info: AgentsSpanHeader): Headers {
  return { ...headers, [AGENTS_TRACE_HEADER_KEY]: defaultPayloadConverter.toPayload(info) };
}

export function extractAgentsTraceHeader(headers: Headers): AgentsSpanHeader | null {
  const payload = headers[AGENTS_TRACE_HEADER_KEY];
  if (!payload) return null;
  return defaultPayloadConverter.fromPayload<AgentsSpanHeader>(payload);
}

export function injectAgentsTraceHeaderNexus(
  headers: Record<string, string>,
  info: AgentsSpanHeader
): Record<string, string> {
  return { ...headers, [AGENTS_TRACE_HEADER_KEY]: JSON.stringify(info) };
}

export function extractAgentsTraceHeaderNexus(headers: Record<string, string>): AgentsSpanHeader | null {
  const raw = headers[AGENTS_TRACE_HEADER_KEY];
  if (!raw) return null;
  try {
    return JSON.parse(raw) as AgentsSpanHeader;
  } catch {
    return null;
  }
}

// Traces the plugin rebuilt on a Worker with the key its `tracingApiKey` option resolved. That key
// is Worker configuration, not the trace's own, so the plugin resolves it again past the next
// boundary.
const tracesWithResolvedKey = new WeakSet<object>();

export function markTracingApiKeyResolved(trace: object): void {
  tracesWithResolvedKey.add(trace);
}

/**
 * Throws if a trace has its own tracing key. Headers are stored in Event History, so they cannot
 * carry the key, and spans past the boundary would be exported with another key. Workflow code
 * is refused where it sets the key; every header write checks again.
 */
export function refuseTracingApiKey(tracingApiKey: string | null | undefined): void {
  if (tracingApiKey == null) return;
  throw new Error(
    "The OpenAI Agents plugin cannot pass a trace's own tracing key, set with tracing: { apiKey } on a run " +
      'or tracingApiKey on a trace, to Temporal Workflows and Activities, so their spans would be exported ' +
      'with another key. Set the tracing key on every Client and Worker process instead, with ' +
      "setTracingExportApiKey() or OPENAI_API_KEY, or resolve it on the Worker with the plugin's tracingApiKey option."
  );
}

export function currentAgentsSpanHeader(): AgentsSpanHeader {
  const ctx = getCurrentTraceContext();
  if (ctx && !tracesWithResolvedKey.has(ctx.trace)) refuseTracingApiKey(ctx.trace.tracingApiKey);
  const otelSpanContext = otel.trace.getSpanContext(otel.context.active());
  return {
    traceName: ctx?.trace.name ?? 'Unknown Workflow',
    spanId: ctx?.span?.spanId ?? null,
    traceId: ctx?.trace.traceId ?? null,
    ...(otelSpanContext ? { otelSpanId: otelSpanContext.spanId } : {}),
  };
}

export const AGENTS_CONFIG_HEADER_KEY = '__openai_agents_config';

export interface AgentsConfigHeader {
  addTemporalSpans?: boolean;
  useOtelInstrumentation?: boolean;
  modelParams?: SerializableModelActivityOptions;
}

export function injectAgentsConfigHeader(headers: Headers, config: AgentsConfigHeader): Headers {
  return { ...headers, [AGENTS_CONFIG_HEADER_KEY]: defaultPayloadConverter.toPayload(config) };
}

export function extractAgentsConfigHeader(headers: Headers): AgentsConfigHeader | undefined {
  const payload = headers[AGENTS_CONFIG_HEADER_KEY];
  if (!payload) return undefined;
  return defaultPayloadConverter.fromPayload<AgentsConfigHeader>(payload);
}
