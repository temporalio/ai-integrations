// Per-Workflow plugin config keyed by `${workflowId}/${runId}`. Set by inbound
// interceptor, read by runner + interceptors, cleared on dispose. runId is
// part of the key because under `reuseV8Context: true` an old run can still
// be in the worker cache (un-disposed) when continueAsNew's new run starts
// using the same V8 isolate — workflowId alone would let the old run's
// dispose wipe the new run's entry.
import { workflowInfo } from '@temporalio/workflow';
import type { SerializableModelOptions } from '../common/model-options';

export interface PluginConfig extends SerializableModelOptions {
  addTemporalSpans: boolean;
  useOtelInstrumentation: boolean;
}

const configByRun = new Map<string, PluginConfig>();

function currentKey(): string {
  const info = workflowInfo();
  return `${info.workflowId}/${info.runId}`;
}

export function setPluginConfig(config: PluginConfig): void {
  configByRun.set(currentKey(), config);
}

export function clearPluginConfig(): void {
  configByRun.delete(currentKey());
}

export function getCurrentPluginConfig(): PluginConfig | undefined {
  return configByRun.get(currentKey());
}
