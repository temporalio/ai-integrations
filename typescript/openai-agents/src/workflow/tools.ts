import type { FunctionTool, RunContext } from '@openai/agents-core';
import { ApplicationFailure, type Duration, type RetryPolicy } from '@temporalio/common';
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

type ActivityInput<TActivity extends (input: any) => Promise<any>> = Parameters<TActivity>[0];

/**
 * Definition for wrapping a Temporal Activity as an agent tool. `TActivity` is
 * the Activity function type (e.g. `typeof activities.getWeather`); its first
 * argument shape types `parameters`.
 */
export interface ActivityToolDefinition<TActivity extends (input: any) => Promise<any>> {
  /** Activity name — must match the registered Activity on the Worker. */
  name: string;
  /** Tool description shown to the model. */
  description: string;
  /** Explicit JSON schema for the tool parameters (not Zod). */
  parameters: JsonObjectSchema<ActivityInput<TActivity>>;
  /**
   * Name of one property in `parameters`. When set, the Activity gets that
   * property's value as its argument instead of the whole arguments object.
   * Use it for an Activity whose parameter is a string, number or other
   * scalar, such as `greet(name: string)`.
   */
  inputProperty?: string;
}

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
export function activityAsTool<TActivity extends (input: any) => Promise<any>>(
  definition: ActivityToolDefinition<TActivity>,
  options?: ActivityAsToolOptions
): FunctionTool {
  const { inputProperty } = definition;
  if (inputProperty !== undefined && !Object.hasOwn(definition.parameters.properties, inputProperty)) {
    throw ApplicationFailure.create({
      message: `activityAsTool('${definition.name}'): inputProperty '${inputProperty}' is not a property in parameters.`,
      type: 'AgentsWorkflowError',
      nonRetryable: true,
    });
  }
  const activities = proxyActivities<Record<string, (input: any) => Promise<any>>>({
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
    invoke: async (_runContext: RunContext<any>, input: string): Promise<string> => {
      let parsedInput: any;
      try {
        parsedInput = JSON.parse(input);
      } catch (e) {
        throw new ToolSerializationError(`Failed to parse tool input for '${definition.name}': ${e}`);
      }
      const activityInput: ActivityInput<TActivity> =
        inputProperty === undefined ? parsedInput : parsedInput?.[inputProperty];
      const result = await activities[definition.name]!(activityInput);
      if (result === undefined) return '';
      return typeof result === 'string' ? result : JSON.stringify(result);
    },
    needsApproval: async () => false,
    isEnabled: async () => true,
  } as FunctionTool;

  return t;
}
