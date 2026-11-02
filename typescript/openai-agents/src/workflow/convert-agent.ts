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
 * An object is only reused if its handoffs lead to the same objects as before, so a state saved on the
 * earlier graph still reaches every agent it reached, and serializes the decisions those agents own.
 * Other agents, and agents whose class overrides `clone`, get their fresh clone. Such a clone can hold
 * state, such as private fields or entries keyed by the clone, that can't be moved onto another object.
 * The objects only change once every agent has cloned, and get their earlier fields back if a
 * `Handoff.clone` then throws, so a conversion that throws leaves them as they were. For each object it
 * changes, the conversion adds a function to `undo` that gives the object back its earlier fields.
 */
export function convertAgent(
  agent: Agent<any, any>,
  modelParams: ModelActivityOptions,
  seen?: Map<Agent<any, any>, Agent<any, any>>,
  modelNameOverride?: string,
  identities?: ReadonlyMap<Agent<any, any>, Agent<any, any>>,
  undo?: (() => void)[]
): Agent<any, any> {
  seen = seen ?? new Map();
  if (seen.has(agent)) return seen.get(agent)!;
  const clones = new Map<Agent<any, any>, Agent<any, any>>();
  cloneAgentGraph(agent, modelParams, seen, clones, modelNameOverride);
  const reused = reusableIdentities(clones, identities);
  for (const [source, clone] of clones) {
    seen.set(source, reused.get(source) ?? clone);
  }
  // Give the reused objects their new fields before any `Handoff.clone` sees them, as a fresh conversion does.
  const undoFields: (() => void)[] = [];
  for (const [source, target] of reused) {
    undoFields.push(replaceFields(target, clones.get(source)!));
  }
  try {
    connectHandoffs(agent, seen, clones, new Set());
  } catch (error) {
    for (const restore of undoFields.reverse()) restore();
    throw error;
  }
  undo?.push(...undoFields);
  return seen.get(agent)!;
}

/**
 * Sets the handoffs of the converted agent of each source agent in `clones` reachable from `agent`. Each
 * handoff target is connected before its `Handoff.clone` runs, so the clone sees the converted target.
 */
function connectHandoffs(
  agent: Agent<any, any>,
  seen: ReadonlyMap<Agent<any, any>, Agent<any, any>>,
  clones: ReadonlyMap<Agent<any, any>, Agent<any, any>>,
  connected: Set<Agent<any, any>>
): void {
  if (!clones.has(agent) || connected.has(agent)) return;
  connected.add(agent);
  seen.get(agent)!.handoffs = (agent.handoffs ?? []).map((h) => {
    const handoffAgent = h instanceof Handoff ? h.agent : h;
    connectHandoffs(handoffAgent, seen, clones, connected);
    const converted = seen.get(handoffAgent)!;
    return h instanceof Handoff ? h.clone({ agent: converted }) : converted;
  });
}

/** Validates and clones each agent in `agent`'s graph that isn't in `seen`, into `clones`. */
function cloneAgentGraph(
  agent: Agent<any, any>,
  modelParams: ModelActivityOptions,
  seen: ReadonlyMap<Agent<any, any>, Agent<any, any>>,
  clones: Map<Agent<any, any>, Agent<any, any>>,
  modelNameOverride: string | undefined
): void {
  if (seen.has(agent) || clones.has(agent)) return;

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

  clones.set(agent, agent.clone({ model: activityBackedModel }));
  for (const h of agent.handoffs ?? []) {
    cloneAgentGraph(h instanceof Handoff ? h.agent : h, modelParams, seen, clones, modelNameOverride);
  }
}

/**
 * The objects in `identities` this conversion reuses, keyed by source agent: those whose fields
 * `canReplaceFields` allows replacing, and whose handoffs, once converted, lead to the same objects as
 * before. Rewiring a reused object to a new one would hide the agent it led to from states saved on the
 * earlier graph, which then lose that agent's approvals and rejections when serialized.
 */
function reusableIdentities(
  clones: ReadonlyMap<Agent<any, any>, Agent<any, any>>,
  identities: ReadonlyMap<Agent<any, any>, Agent<any, any>> | undefined
): Map<Agent<any, any>, Agent<any, any>> {
  const reused = new Map<Agent<any, any>, Agent<any, any>>();
  for (const [source, clone] of clones) {
    const identity = identities?.get(source);
    if (identity !== undefined && canReplaceFields(source, identity, clone)) reused.set(source, identity);
  }
  // Dropping an agent can change the targets of agents that hand off to it, so repeat until none changes.
  let dropped = true;
  while (dropped) {
    dropped = false;
    for (const [source, identity] of reused) {
      const before = new Set(handoffAgents(identity));
      const after = new Set(handoffAgents(source).map((target) => reused.get(target)));
      if (after.size !== before.size || [...after].some((target) => target === undefined || !before.has(target))) {
        reused.delete(source);
        dropped = true;
      }
    }
  }
  return reused;
}

function handoffAgents(agent: Agent<any, any>): Agent<any, any>[] {
  return (agent.handoffs ?? []).map((h) => (h instanceof Handoff ? h.agent : h));
}

/**
 * Whether `target` can take the fields of `clone`, which `agent.clone` returned. `Agent.clone` and
 * `SandboxAgent.clone` return instances of exactly those classes, which keep all their state in own
 * properties. Any other `clone`, even one that returns such an instance, can keep state elsewhere.
 */
function canReplaceFields(agent: Agent<any, any>, target: Agent<any, any>, clone: Agent<any, any>): boolean {
  const plainPrototypes: readonly object[] = [Agent.prototype, SandboxAgent.prototype];
  return (
    (agent.clone === Agent.prototype.clone || agent.clone === SandboxAgent.prototype.clone) &&
    plainPrototypes.includes(Object.getPrototypeOf(target)) &&
    plainPrototypes.includes(Object.getPrototypeOf(clone))
  );
}

/**
 * Gives `target` exactly the own properties and prototype of `source`, and returns a function that gives
 * `target` back the ones it had before.
 */
function replaceFields(target: Agent<any, any>, source: Agent<any, any>): () => void {
  const descriptors = Object.getOwnPropertyDescriptors(target);
  const prototype = Object.getPrototypeOf(target);
  setFields(target, Object.getOwnPropertyDescriptors(source), Object.getPrototypeOf(source));
  return () => setFields(target, descriptors, prototype);
}

function setFields(target: Agent<any, any>, descriptors: PropertyDescriptorMap, prototype: object): void {
  for (const key of Reflect.ownKeys(target)) {
    delete (target as any)[key];
  }
  Object.defineProperties(target, descriptors);
  if (Object.getPrototypeOf(target) !== prototype) {
    Object.setPrototypeOf(target, prototype);
  }
}
