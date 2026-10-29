import { Agent, Handoff, MemorySession, Runner, type RunContext, type Tool } from '@openai/agents-core';
import type { SandboxRunConfig } from '@openai/agents-core/sandbox';
import { ApplicationFailure } from '@temporalio/common';
import { inWorkflowContext } from '@temporalio/workflow';
import { ActivityBackedModelProvider } from './activity-backed-model';
import { type AgentToolRecord, checkAgentToolOptions, getAgentToolRecord, isUnrecordedAgentTool } from './agent-tools';
import { validateTemporalSandboxClient } from './sandbox-client';

/**
 * Checks an agent graph before a run, without changing it: the agents reached through handoffs and
 * through `agent.asTool()` tools, and those tools' options. Rejects raw functions in `tools`, an agent
 * with no model when the run sets none, and setups that would run model calls or sandbox operations
 * in the Workflow. Agents then run as written: the runner's model provider sends every model name to
 * the model Activity. Uses a seen-set for circular references.
 */
export function checkAgent(agent: Agent<any, any>, modelOverride?: string, seen = new Set<Agent<any, any>>()): void {
  if (seen.has(agent)) return;
  seen.add(agent);

  checkAgentCopy(agent);
  rejectModelObject(agent);
  if (typeof agent.model !== 'string' && modelOverride === undefined) {
    throw ApplicationFailure.create({
      message:
        `Agent '${agent.name}' has no model declared and no runConfig.model override given. ` +
        'Declare a model on the agent or pass runConfig.model to runner.run().',
      type: 'AgentsWorkflowError',
      nonRetryable: true,
    });
  }
  for (const tool of agent.tools ?? []) checkTool(tool, agent, modelOverride, seen);
  for (const h of agent.handoffs ?? []) checkAgent(h instanceof Handoff ? h.agent : h, modelOverride, seen);
}

function checkTool(
  tool: Tool<any>,
  owner: Agent<any, any>,
  modelOverride: string | undefined,
  seen: Set<Agent<any, any>>
): void {
  const record = checkToolOptions(tool, owner);
  if (record === undefined) return;
  const nestedModel = typeof record.runConfig?.model === 'string' ? record.runConfig.model : modelOverride;
  checkAgent(record.agent, nestedModel, seen);
}

interface RunningAgentCheckState {
  original: Agent<any, any>['getSystemPrompt'];
}

const runningAgentCheckState = Symbol.for('@temporalio/openai-agents/runningAgentCheck');

/**
 * Checks each agent again just before it calls its model, in the Workflow: an agent that a custom
 * `Handoff`'s callback returns, or that an `agent.on(...)` listener changes, isn't in the graph
 * `checkAgent` sees before the run. Upstream calls `getSystemPrompt` once per model call, after it
 * picks the model and before it calls it, so the wrapper rejects a `Model` object and unsafe
 * `asTool()` tools there. The agents behind those tools are checked when they run, and the config
 * their nested runs use when those runs start.
 *
 * `prototype` is the `Agent` prototype of the `@openai/agents-core` copy to check. The runner calls
 * this for each agent its runs reach (see `installRunnerCheck`), so agents from another copy, such
 * as a dependency's, are checked too.
 */
export function installRunningAgentCheck(prototype: object = Agent.prototype): void {
  const target = prototype as Agent<any, any> & Record<symbol, RunningAgentCheckState | undefined>;
  if (target[runningAgentCheckState] !== undefined) return;
  const state: RunningAgentCheckState = { original: target.getSystemPrompt };
  function getSystemPrompt(this: Agent<any, any>, runContext: RunContext<any>): Promise<string | undefined> {
    if (inWorkflowContext()) {
      rejectModelObject(this);
      for (const tool of this.tools ?? []) checkToolOptions(tool, this);
    }
    return state.original.call(this, runContext);
  }
  Object.defineProperty(target, runningAgentCheckState, { value: state });
  target.getSystemPrompt = getSystemPrompt;
}

installRunningAgentCheck();

/**
 * Installs the check before each model call on the `Agent` class of `agent`'s copy of
 * `@openai/agents-core`: the furthest prototype that defines `getSystemPrompt`, so subclasses that
 * call `super` are covered.
 */
function checkAgentCopy(agent: Agent<any, any>): void {
  let base: object | undefined;
  for (let p = Object.getPrototypeOf(agent); p !== null; p = Object.getPrototypeOf(p)) {
    if (Object.prototype.hasOwnProperty.call(p, 'getSystemPrompt')) base = p;
  }
  if (base === undefined) {
    throw ApplicationFailure.create({
      message: `Agent '${agent.name}' is not an @openai/agents-core Agent, so it cannot be checked before it calls its model.`,
      type: 'AgentsWorkflowError',
      nonRetryable: true,
    });
  }
  installRunningAgentCheck(base);
}

interface RunnerCheckState {
  original: (this: Runner, agent: Agent<any, any>, input: unknown, options?: RunOptionsLike) => Promise<unknown>;
  watched: WeakSet<Runner>;
}

interface RunOptionsLike {
  session?: unknown;
  sandbox?: SandboxRunConfig;
}

const runnerCheckState = Symbol.for('@temporalio/openai-agents/runnerCheck');

/**
 * Checks what each `Runner` run in the Workflow uses, as it starts: the runner's own, and the one
 * upstream builds for each `agent.asTool()` call. Upstream builds that nested config only then, after
 * the tool's schema, its `inputBuilder` and any listener ran, and spreads the tool's `runConfig` over
 * what it inherits, so this is the one place that sees the provider, model, session and sandbox the
 * run uses. The model provider must be the runner's, which sends model calls to Activities. The run
 * gets a frozen copy of its sandbox config, which upstream reads as it goes, so a later change to the
 * caller's object, such as an `agent_start` listener's, doesn't reach it.
 *
 * Upstream emits `agent_start` on the runner before each agent's first model call, including an
 * agent that a custom `Handoff`'s callback returns, so the run installs the check before each model
 * call on the copy of `@openai/agents-core` of every agent it starts.
 */
export function installRunnerCheck(): void {
  const prototype = Runner.prototype as unknown as Record<symbol, RunnerCheckState | undefined>;
  if (prototype[runnerCheckState] !== undefined) return;
  const state: RunnerCheckState = {
    original: Runner.prototype.run as RunnerCheckState['original'],
    watched: new WeakSet(),
  };
  async function run(this: Runner, agent: Agent<any, any>, input: unknown, options?: RunOptionsLike) {
    if (!inWorkflowContext()) return state.original.call(this, agent, input, options);
    if (!state.watched.has(this)) {
      state.watched.add(this);
      this.on('agent_start', (_context, started) => checkAgentCopy(started));
    }
    checkAgentCopy(agent);
    const where = `The run of agent '${agent.name}'`;
    if (!(this.config.modelProvider instanceof ActivityBackedModelProvider)) {
      throw ApplicationFailure.create({
        message:
          `${where} has a model provider that would call models in the Workflow. Run agents with ` +
          "TemporalOpenAIRunner, and don't set or clear runConfig.modelProvider in agent.asTool().",
        type: 'AgentsWorkflowError',
        nonRetryable: true,
      });
    }
    const { model } = this.config;
    if (model !== undefined && typeof model !== 'string') {
      throw ApplicationFailure.create({
        message:
          `${where} has a Model object as runConfig.model. In Temporal workflows, all models must be ` +
          'specified as strings.',
        type: 'AgentsWorkflowError',
        nonRetryable: true,
      });
    }
    if (options?.session instanceof MemorySession) {
      throw ApplicationFailure.create({
        message:
          `${where} has a MemorySession, which is not safe inside a Workflow. ` +
          'Pass WorkflowSafeMemorySession from @temporalio/openai-agents/workflow.',
        type: 'UnsafeSessionError',
        nonRetryable: true,
      });
    }
    // Upstream uses `options.sandbox ?? config.sandbox`, and passes the same to nested runs.
    const sandbox = options?.sandbox ?? this.config.sandbox;
    if (sandbox === undefined) return state.original.call(this, agent, input, options);
    validateTemporalSandboxClient(
      sandbox,
      `${where} ${options?.sandbox !== undefined ? 'runOptions.sandbox' : 'runConfig.sandbox'}`
    );
    return state.original.call(this, agent, input, { ...options, sandbox: Object.freeze({ ...sandbox }) });
  }
  Object.defineProperty(Runner.prototype, runnerCheckState, { value: state });
  Runner.prototype.run = run as typeof Runner.prototype.run;
}

installRunnerCheck();

/**
 * Rejects a raw function, an unrecorded `asTool()` tool, and `asTool()` options that would run in the
 * Workflow. Returns the record of an `asTool()` tool, so the caller can check its agent.
 */
function checkToolOptions(tool: Tool<any>, owner: Agent<any, any>): AgentToolRecord | undefined {
  // Reject raw functions: authoring mistake — use tool() or activityAsTool().
  if (typeof tool === 'function') {
    throw ApplicationFailure.create({
      message:
        `Agent '${owner.name}': Provided tool is a raw function, not a tool object. ` +
        'Did you mean to use tool() or activityAsTool()?',
      type: 'AgentsWorkflowError',
      nonRetryable: true,
    });
  }
  const record = getAgentToolRecord(tool);
  if (record === undefined) {
    if (isUnrecordedAgentTool(tool)) {
      throw ApplicationFailure.create({
        message:
          `Agent '${owner.name}': tool '${tool.name}' looks like an agent.asTool() tool that ` +
          '@temporalio/openai-agents did not record, so its nested agent cannot be checked. ' +
          'Create it with agent.asTool() in Workflow code, after @temporalio/openai-agents/workflow ' +
          'is loaded and with the same @openai/agents-core copy, and pass it as is.',
        type: 'AgentsWorkflowError',
        nonRetryable: true,
      });
    }
    return undefined;
  }
  checkAgentToolOptions(record, tool.name);
  return record;
}

function rejectModelObject(agent: Agent<any, any>): void {
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
}
