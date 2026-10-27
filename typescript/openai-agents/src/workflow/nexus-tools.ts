import type { FunctionTool, RunContext } from '@openai/agents-core';
import type * as nexus from 'nexus-rpc';
import { ApplicationFailure, type Duration } from '@temporalio/common';
import { createNexusServiceClient } from '@temporalio/workflow';
import { ToolSerializationError, type JsonObjectSchema } from './tools';

/** `TOp`'s input shape types `parameters`. */
export interface NexusOperationToolDefinition<TOp extends nexus.OperationDefinition<any, any>> {
  name: string;
  description: string;
  parameters: JsonObjectSchema<nexus.OperationInput<TOp>>;
  /**
   * Name of one property in `parameters`. When set, the Operation gets that
   * property's value as its input instead of the whole arguments object.
   * Use it for an Operation whose input is a string, number or other scalar,
   * such as `nexus.operation<string, string>()`.
   */
  inputProperty?: string;
}

export interface NexusOperationAsToolOptions<TService extends nexus.ServiceDefinition> {
  service: TService;
  endpoint: string;
  scheduleToCloseTimeout?: Duration;
  strict?: boolean;
}

/**
 * Wraps a Nexus Operation as an OpenAI Agents `FunctionTool`. When the agent
 * invokes the tool, it calls the Operation through a Workflow-side Nexus
 * client (`createNexusServiceClient`) and returns the stringified result.
 *
 * This is the Nexus counterpart of {@link activityAsTool}: instead of
 * scheduling a Temporal Activity, the tool invocation schedules a Nexus
 * Operation against the configured endpoint and service.
 */
export function nexusOperationAsTool<
  TService extends nexus.ServiceDefinition,
  TOp extends TService['operations'][keyof TService['operations']],
>(
  operation: TOp,
  definition: NexusOperationToolDefinition<TOp>,
  options: NexusOperationAsToolOptions<TService>
): FunctionTool {
  const { inputProperty } = definition;
  if (inputProperty !== undefined && !Object.hasOwn(definition.parameters.properties, inputProperty)) {
    throw ApplicationFailure.create({
      message: `nexusOperationAsTool('${definition.name}'): inputProperty '${inputProperty}' is not a property in parameters.`,
      type: 'AgentsWorkflowError',
      nonRetryable: true,
    });
  }
  const tool = {
    type: 'function',
    name: definition.name,
    description: definition.description,
    parameters: definition.parameters as any,
    strict: options.strict ?? true,
    invoke: async (_runContext: RunContext<any>, input: string): Promise<string> => {
      let parsedInput: any;
      try {
        parsedInput = JSON.parse(input);
      } catch (e) {
        throw new ToolSerializationError(`Failed to parse tool input for '${definition.name}': ${e}`);
      }
      const client = createNexusServiceClient<TService>({
        service: options.service,
        endpoint: options.endpoint,
      });
      const operationInput: nexus.OperationInput<TOp> =
        inputProperty === undefined ? parsedInput : parsedInput?.[inputProperty];
      const result = await client.executeOperation(operation, operationInput, {
        scheduleToCloseTimeout: options.scheduleToCloseTimeout,
      });
      if (result === undefined) return '';
      return typeof result === 'string' ? result : JSON.stringify(result);
    },
    needsApproval: async () => false,
    isEnabled: async () => true,
  } as FunctionTool;

  return tool;
}
