import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { helpers } from '@temporalio/test-helpers';
import { Worker, bundleWorkflowCode } from '@temporalio/worker';
import { OpenAIAgentsPlugin } from '..';
import { runningAgentRejectedWorkflow, type RunningAgentRejection } from './workflows/openai-agents';
import type { AgentCopyCase } from './workflows/openai-agents-copies';
import { FakeModelProvider, textResponse, toolCallResponse, type FakeModel } from './stubs/openai-agents';
import { handoffResponse } from './stubs/openai-agents-fakes';
import { makeOpenAIAgentsTestFunction, replay } from './helpers/openai-agents';

const test = makeOpenAIAgentsTestFunction();

test('A Workflow module with ES module imports loads with one copy of @openai/agents-core', async (t) => {
  const { createWorker, taskQueue } = helpers(t);
  const plugin = new OpenAIAgentsPlugin({
    modelProvider: new FakeModelProvider([textResponse('ESM module response')]),
  });
  // Bundle the ES module source, as the Worker does for a `.ts` Workflow file run from source.
  const workflowBundle = await bundleWorkflowCode({
    workflowsPath: require.resolve('../../src/__tests__/workflows/openai-agents-esm.mjs'),
    plugins: [plugin],
  });
  const esModuleBuild = new Set(workflowBundle.code.match(/@openai[\\/]agents-core[\\/]dist[\\/][^"'\s]*\.mjs/g));
  t.deepEqual([...esModuleBuild].slice(0, 3), [], `${esModuleBuild.size} modules from the ES module build`);

  const worker = await createWorker({ workflowBundle, plugins: [plugin] });
  const result = await worker.runUntil(
    t.context.env.client.workflow.execute('esmModuleAgentWorkflow', {
      taskQueue,
      workflowId: `esm-module-${Date.now()}`,
      args: ['Hello'],
    })
  );
  t.is(result, 'ESM module response');
});

test('A Workflow module with ES module imports bundles one copy of @openai/agents-core with async chunks off', async (t) => {
  const plugin = new OpenAIAgentsPlugin({ modelProvider: new FakeModelProvider([]) });
  // These configs avoid the async chunk, so a bundle with both builds of @openai/agents-core would
  // still load, and the plugin's instanceof checks would miss objects from one of them.
  const setups: Record<string, NonNullable<Parameters<typeof bundleWorkflowCode>[0]['webpackConfigHook']>> = {
    'output.asyncChunks: false': (config) => ({ ...config, output: { ...config.output, asyncChunks: false } }),
    "dynamicImportMode: 'eager'": (config) => ({
      ...config,
      module: { ...config.module, parser: { ...config.module?.parser, javascript: { dynamicImportMode: 'eager' } } },
    }),
  };
  for (const [name, webpackConfigHook] of Object.entries(setups)) {
    const { code } = await bundleWorkflowCode({
      workflowsPath: require.resolve('../../src/__tests__/workflows/openai-agents-esm.mjs'),
      plugins: [plugin],
      webpackConfigHook,
    });
    const esModuleBuild = new Set(code.match(/@openai[\\/]agents-core[\\/]dist[\\/][^"'\s]*\.mjs/g));
    t.deepEqual([...esModuleBuild].slice(0, 3), [], `${name}: ${esModuleBuild.size} modules from the ES module build`);
  }
});

test('Agents picked or changed during the run are checked before they call their model', async (t) => {
  const { createWorker, startWorkflow } = helpers(t);
  const models: string[] = [];
  const provider = {
    getModel(model: string) {
      models.push(model);
      return {
        async getResponse(request: any) {
          if (request.systemInstructions === 'Use the tool.') return toolCallResponse('ask_nested', { input: 'q' });
          if (request.systemInstructions === 'Answer as nested.') return textResponse('nested answer');
          return handoffResponse('transfer_to_desk');
        },
      } as unknown as FakeModel;
    },
  };
  const worker = await createWorker({ plugins: [new OpenAIAgentsPlugin({ modelProvider: provider })] });
  // [expected result, model Activities the run schedules]
  const expected: Record<RunningAgentRejection, [RegExp, number]> = {
    'handoff-callback-model-object': [/^AgentsWorkflowError: .*Agent 'Picked' has a Model object/, 1],
    'handoff-callback-model-provider': [
      /^AgentsWorkflowError: .*asTool\('ask_nested'\) sets runConfig.modelProvider/,
      1,
    ],
    'agent-start-model-object': [/^AgentsWorkflowError: .*Agent 'Lead' has a Model object/, 0],
    // A nested run's failure reaches the lead only as upstream's generic tool output, and the run ends on it.
    // A missing check shows as the inline model's answer in the output or as one more model Activity.
    'nested-agent-start-model-object': [/^An error occurred while running the tool/, 1],
    // These change the tool's config after the lead's model call, so the nested run's own check catches them.
    'tool-start-model-provider': [/^An error occurred while running the tool/, 1],
    'tool-start-run-model-object': [/^An error occurred while running the tool/, 1],
    'input-builder-model-provider': [/^An error occurred while running the tool/, 1],
    'tool-start-clear-model-provider': [/^An error occurred while running the tool/, 1],
    'tool-start-memory-session': [/^An error occurred while running the tool/, 1],
    'tool-start-tracing-key': [/^An error occurred while running the tool/, 1],
    'tool-start-run-config-tracing-key': [/^An error occurred while running the tool/, 1],
    'tool-option-model-provider-undefined': [
      /^AgentsWorkflowError: .*asTool\('ask_nested'\) sets runConfig.modelProvider/,
      0,
    ],
    'tool-start-run-model-name': [/^nested answer$/, 2],
  };

  await worker.runUntil(async () => {
    for (const [kind, [result, modelActivities]] of Object.entries(expected)) {
      const handle = await startWorkflow(runningAgentRejectedWorkflow, {
        args: [kind as RunningAgentRejection],
        workflowExecutionTimeout: '30 seconds',
      });
      const output = await handle.result();
      t.regex(output, result, kind);
      t.notRegex(output, /the inline model ran/, kind);
      const { events } = await handle.fetchHistory();
      const scheduled = (events ?? []).filter((e) => e.activityTaskScheduledEventAttributes != null);
      t.is(scheduled.length, modelActivities, `${kind}: model Activities`);
      await replay(t, handle);
    }
  });
  // The control's change to the tool's runConfig.model reaches the nested model Activity.
  t.true(models.includes('changed-run-model'), models.join(', '));
  t.false(models.includes('nested-run-model'), models.join(', '));
});

test('Agents from another copy of @openai/agents-core are checked before they call their model', async (t) => {
  const { createWorker, taskQueue } = helpers(t);
  // Bundle a copy of the package under another name, as a dependency that ships its own copy would.
  const packageDir = fs.realpathSync(path.dirname(path.dirname(require.resolve('@openai/agents-core'))));
  const copyDir = fs.mkdtempSync(path.join(os.tmpdir(), 'agents-core-copy-'));
  t.teardown(() => fs.rmSync(copyDir, { recursive: true, force: true }));
  fs.cpSync(packageDir, copyDir, { recursive: true, filter: (src) => !/\.(d\.ts|map)$/.test(src) });
  const provider = {
    getModel() {
      return {
        async getResponse(request: any) {
          if (request.systemInstructions === 'Route.') return handoffResponse('transfer_to_desk');
          if (request.systemInstructions === 'Switch.') return toolCallResponse('switch_model', {});
          if (request.systemInstructions === 'Use the tool.') return toolCallResponse('ask_nested', { input: 'q' });
          if (request.systemInstructions === 'Answer as nested.') return textResponse('nested answer');
          if (request.systemInstructions === 'Answer as declared.') return textResponse('declared answer');
          return textResponse('copy answer');
        },
      } as unknown as FakeModel;
    },
  };
  const plugin = new OpenAIAgentsPlugin({ modelProvider: provider });
  const workflowBundle = await bundleWorkflowCode({
    workflowsPath: require.resolve('./workflows/openai-agents-copies'),
    plugins: [plugin],
    webpackConfigHook: (config) => ({
      ...config,
      resolve: {
        ...config.resolve,
        alias: { ...(config.resolve?.alias as object), 'agents-core-copy$': path.join(copyDir, 'dist', 'index.js') },
        // The copy's own dependencies, such as `debug` and `zod`.
        modules: [...(config.resolve?.modules ?? ['node_modules']), path.dirname(path.dirname(packageDir))],
      },
    }),
  });
  const worker = await createWorker({ workflowBundle, plugins: [plugin] });
  // [expected result, model Activities the run schedules]
  const expected: Record<AgentCopyCase, [RegExp, number]> = {
    'agent-start-model-object': [/^AgentsWorkflowError: .*Agent 'Lead' has a Model object/, 0],
    'handoff-callback-model-object': [/^AgentsWorkflowError: .*Agent 'Picked' has a Model object/, 1],
    'tool-model-object': [/^AgentsWorkflowError: .*Agent 'Lead' has a Model object/, 1],
    'as-tool': [/^AgentsWorkflowError: .*tool 'ask_nested' looks like an agent.asTool\(\) tool/, 0],
    'deprecated-alias': [/^nested answer$/, 2],
    'deprecated-alias-override': [/^AgentsWorkflowError: .*tool 'ask_nested' looks like an agent.asTool\(\) tool/, 0],
    'string-model': [/^copy answer$/, 1],
  };
  await worker.runUntil(async () => {
    for (const [kind, [result, modelActivities]] of Object.entries(expected)) {
      // By name: the Workflow module needs the bundle's `agents-core-copy`.
      const handle = await t.context.env.client.workflow.start('agentCopyWorkflow', {
        taskQueue,
        workflowId: `agent-copy-${kind}-${Date.now()}`,
        args: [kind],
        workflowExecutionTimeout: '30 seconds',
      });
      const output: string = await handle.result();
      t.regex(output, result, kind);
      t.notRegex(output, /the inline model ran/, kind);
      const { events } = await handle.fetchHistory();
      const scheduled = (events ?? []).filter((e) => e.activityTaskScheduledEventAttributes != null);
      t.is(scheduled.length, modelActivities, `${kind}: model Activities`);
      await Worker.runReplayHistory(
        { workflowBundle, plugins: [new OpenAIAgentsPlugin({ modelProvider: new FakeModelProvider([]) })] },
        await handle.fetchHistory()
      );
    }
  });
});
