import { Agent, Handoff } from '@openai/agents-core';
import { SandboxAgent } from '@openai/agents-core/sandbox';
import { ApplicationFailure } from '@temporalio/common';
import type { ModelActivityOptions } from '../common/model-activity-options';
import { ActivityBackedModel } from './activity-backed-model';

/**
 * Recursively converts an agent graph: validate tools, replace each agent's
 * model with an ActivityBackedModel, and clone handoffs. Uses a seen-map to
 * handle circular handoff references.
 *
 * When `identities` maps a source agent to an agent an earlier conversion returned, that object is
 * returned again, with the fields of this conversion's clone. Each agent then has the same fields as
 * after a fresh conversion, but keeps its object identity, which approvals on a `RunContext` are keyed by.
 * Those objects only change once the whole graph has converted, so a conversion that throws leaves them
 * as they were. An agent whose class overrides `clone` gets its fresh clone instead, since its clone can
 * hold state, such as private fields, that can't be moved onto another object.
 */
export function convertAgent(
  agent: Agent<any, any>,
  modelParams: ModelActivityOptions,
  seen?: Map<Agent<any, any>, Agent<any, any>>,
  modelNameOverride?: string,
  identities?: ReadonlyMap<Agent<any, any>, Agent<any, any>>
): Agent<any, any> {
  const replacements = new Map<Agent<any, any>, Agent<any, any>>();
  const converted = cloneAgentGraph(
    agent,
    modelParams,
    seen ?? new Map(),
    new Map(),
    replacements,
    modelNameOverride,
    identities
  );
  for (const [target, clone] of replacements) {
    replaceFields(target, clone);
  }
  return converted;
}

/**
 * Converts `agent`'s graph like `convertAgent`, without changing the objects in `identities`. Each reused
 * object is added to `replacements`, mapped to the clone whose fields it gets.
 */
function cloneAgentGraph(
  agent: Agent<any, any>,
  modelParams: ModelActivityOptions,
  seen: Map<Agent<any, any>, Agent<any, any>>,
  clones: Map<Agent<any, any>, Agent<any, any>>,
  replacements: Map<Agent<any, any>, Agent<any, any>>,
  modelNameOverride: string | undefined,
  identities: ReadonlyMap<Agent<any, any>, Agent<any, any>> | undefined
): Agent<any, any> {
  if (seen.has(agent)) return seen.get(agent)!;

  // Reject raw functions: authoring mistake — use tool() or activityAsTool().
  const tools = agent.tools ?? [];
  for (const t of tools) {
    if (typeof t === 'function') {
      throw ApplicationFailure.create({
        message:
          `Agent '${agent.name}': Provided tool is a raw function, not a tool object. ` +
          'Did you mean to use tool() or activityAsTool()?',
        type: 'AgentsWorkflowError',
        nonRetryable: true,
      });
    }
  }

  const rawModel = agent.model;
  if (rawModel !== undefined && rawModel !== null && typeof rawModel !== 'string') {
    throw ApplicationFailure.create({
      message:
        `Agent '${agent.name}' has a Model object instead of a string model name. ` +
        'In Temporal workflows, all models must be specified as strings — use ' +
        'runConfig.model to override, or declare a string model on the agent.',
      type: 'AgentsWorkflowError',
      nonRetryable: true,
    });
  }
  const modelName = modelNameOverride ?? (typeof rawModel === 'string' ? rawModel : undefined);
  if (modelName === undefined) {
    throw ApplicationFailure.create({
      message:
        `Agent '${agent.name}' has no model declared and no runConfig.model override given. ` +
        'Declare a model on the agent or pass runConfig.model to runner.run().',
      type: 'AgentsWorkflowError',
      nonRetryable: true,
    });
  }
  const activityBackedModel = new ActivityBackedModel(modelName, modelParams);
  // Bind to the original (pre-clone) agent so the summary provider sees the
  // user-declared name and instructions.
  activityBackedModel.setAgent(agent);

  const clone = agent.clone({ model: activityBackedModel });
  const identity = identities?.get(agent);
  const converted = identity !== undefined && canReplaceFields(identity, clone) ? identity : clone;
  seen.set(agent, converted);
  clones.set(agent, clone);

  const convertHandoffAgent = (handoffAgent: Agent<any, any>) =>
    cloneAgentGraph(handoffAgent, modelParams, seen, clones, replacements, modelNameOverride, identities);
  clone.handoffs = (agent.handoffs ?? []).map((h) => {
    if (h instanceof Handoff) {
      const convertedHandoffAgent = convertHandoffAgent(h.agent);
      // A reused target still has its earlier fields here, so take the name from its new clone.
      return h.clone({ agent: convertedHandoffAgent, agentName: clones.get(h.agent)!.name });
    }
    return convertHandoffAgent(h);
  });

  if (converted !== clone) replacements.set(converted, clone);
  return converted;
}

/**
 * Whether `target` can take `source`'s fields. `Agent.clone` and `SandboxAgent.clone` return instances of
 * exactly those classes, which keep all their state in own properties.
 */
function canReplaceFields(target: Agent<any, any>, source: Agent<any, any>): boolean {
  const plainPrototypes: readonly object[] = [Agent.prototype, SandboxAgent.prototype];
  return (
    plainPrototypes.includes(Object.getPrototypeOf(target)) && plainPrototypes.includes(Object.getPrototypeOf(source))
  );
}

/** Gives `target` exactly the own properties and prototype of `source`, and returns `target`. */
function replaceFields(target: Agent<any, any>, source: Agent<any, any>): Agent<any, any> {
  for (const key of Reflect.ownKeys(target)) {
    delete (target as any)[key];
  }
  Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
  if (Object.getPrototypeOf(target) !== Object.getPrototypeOf(source)) {
    Object.setPrototypeOf(target, Object.getPrototypeOf(source));
  }
  return target;
}
