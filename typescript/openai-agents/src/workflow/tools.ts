import type { FunctionTool, RunContext } from '@openai/agents-core';
import type { Duration, RetryPolicy } from '@temporalio/common';
import { proxyActivities } from '@temporalio/workflow';

export class ToolSerializationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ToolSerializationError';
  }
}

export interface JsonObjectSchema<_T = unknown> {
  type: 'object';
  properties: Record<string, unknown>;
  required?: string[];
  additionalProperties?: boolean;
}

/** An Activity that takes the tool input and, optionally, a context copy. */
type ToolActivity = (input: any, context?: any) => Promise<any>;

type ActivityInput<TActivity extends ToolActivity> = Parameters<TActivity>[0];

/**
 * Definition for wrapping a Temporal Activity as an agent tool. `TActivity` is
 * the Activity function type (e.g. `typeof activities.getWeather`); its first
 * argument shape types `parameters`. An Activity with a required second
 * parameter needs `includeContext: true`.
 */
export type ActivityToolDefinition<TActivity extends ToolActivity> = {
  /** Activity name — must match the registered Activity on the Worker. */
  name: string;
  /** Tool description shown to the model. */
  description: string;
  /** Explicit JSON schema for the tool parameters (not Zod). */
  parameters: JsonObjectSchema<ActivityInput<TActivity>>;
} & (TActivity extends (input: any) => Promise<any>
  ? {
      /**
       * When `true`, the Activity gets a copy of the run's context
       * (`runContext.context`) as its second argument. The copy is part of the
       * Activity's input, so it is stored in the Workflow's History, and changes
       * the Activity makes to it do not reach the run.
       */
      includeContext?: boolean;
    }
  : {
      /** Required because the Activity has a required second parameter for the context copy. */
      includeContext: true;
    });

export interface ActivityAsToolOptions {
  startToCloseTimeout?: Duration;
  heartbeatTimeout?: Duration;
  taskQueue?: string;
  retryPolicy?: RetryPolicy;
  strict?: boolean;
}

/**
 * Wraps a Temporal Activity as an OpenAI Agents FunctionTool. When the agent
 * invokes the tool, it schedules the named Activity via `proxyActivities` and
 * returns the stringified result.
 */
export function activityAsTool<TActivity extends ToolActivity>(
  definition: ActivityToolDefinition<TActivity>,
  options?: ActivityAsToolOptions
): FunctionTool;
export function activityAsTool(
  definition: ActivityToolDefinition<(input: any) => Promise<any>>,
  options?: ActivityAsToolOptions
): FunctionTool {
  const activities = proxyActivities<Record<string, (...args: any[]) => Promise<any>>>({
    startToCloseTimeout: options?.startToCloseTimeout ?? '1 minute',
    heartbeatTimeout: options?.heartbeatTimeout,
    taskQueue: options?.taskQueue,
    retry: options?.retryPolicy,
  });

  const t = {
    type: 'function',
    name: definition.name,
    description: definition.description,
    parameters: definition.parameters as any,
    strict: options?.strict ?? true,
    invoke: async (runContext: RunContext<any>, input: string): Promise<string> => {
      let parsedInput: unknown;
      try {
        parsedInput = JSON.parse(input);
      } catch (e) {
        throw new ToolSerializationError(`Failed to parse tool input for '${definition.name}': ${e}`);
      }
      const args = definition.includeContext ? [parsedInput, runContext.context] : [parsedInput];
      const result = await activities[definition.name]!(...args);
      if (result === undefined) return '';
      return typeof result === 'string' ? result : JSON.stringify(result);
    },
    needsApproval: async () => false,
    isEnabled: async () => true,
  } as FunctionTool;

  return t;
}
