// Experimental consumer of the proposed upstream runtime-policy hooks.
import { MemorySession, type Agent, type Model, type Runner } from '@openai/agents-core';
// Proposed public API, supplied by prepare.py in the disposable dependency.
import { setRuntimePolicy } from '@openai/agents-core/runtime-policy';
import { ApplicationFailure } from '@temporalio/common';
import { inWorkflowContext } from '@temporalio/workflow';
import { ActivityBackedModelProvider } from './activity-backed-model';
import { validateTemporalSandboxClient } from './sandbox-client';

function reject(message: string): never {
  throw ApplicationFailure.create({
    message: `Runtime policy: ${message}`,
    type: 'AgentsWorkflowError',
    nonRetryable: true,
  });
}

function checkProvider(runner: Runner): void {
  if (!(runner.config.modelProvider instanceof ActivityBackedModelProvider)) {
    reject('the effective model provider must dispatch Activities');
  }
}

setRuntimePolicy(Object.freeze({
    prepareRun(runner: Runner, _agent: Agent<any, any>, options: any) {
      if (!inWorkflowContext()) return options;
      checkProvider(runner);
      if (runner.config.model !== undefined && typeof runner.config.model !== 'string') {
        reject('runConfig.model must be a name');
      }
      if (options?.session instanceof MemorySession) reject('use WorkflowSafeMemorySession');
      const sandbox = options?.sandbox ?? runner.config.sandbox;
      if (sandbox === undefined) return options;
      validateTemporalSandboxClient(sandbox, options?.sandbox !== undefined ? 'runOptions.sandbox' : 'runConfig.sandbox');
      return { ...options, sandbox: Object.freeze({ ...sandbox }) };
    },
    async resolveModel(runner: Runner, agent: Agent<any, any>, selected: string | Model) {
      if (!inWorkflowContext()) {
        return typeof selected === 'string' ? runner.config.modelProvider.getModel(selected) : selected;
      }
      checkProvider(runner);
      if (typeof selected !== 'string') reject(`agent '${agent.name}' selected a Model object`);
      if (agent.tools.some((tool) => typeof tool === 'function')) reject('use tool() or activityAsTool()');
      return runner.config.modelProvider.getModel(selected);
    },
}));
