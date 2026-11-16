import { Agent, RunState, tool } from '@openai/agents-core';
import { ApplicationFailure, sleep } from '@temporalio/workflow';
import { WorkflowStream } from '@temporalio/workflow-streams/workflow';
import { TemporalModel, TemporalOpenAIRunner, type TemporalModelOptions } from '../../workflow';

export async function temporalModelConfigurationWorkflow(): Promise<boolean> {
  const options: TemporalModelOptions = {
    activityOptions: { startToCloseTimeout: '20m' },
    summaryProvider: { provide: (agent) => `model:${agent?.name}` },
  };
  const model = new TemporalModel('own-model', options);
  options.activityOptions!.startToCloseTimeout = '1ms';
  const agent = new Agent<{ user: string }>({ name: 'Owned', model });
  const clone = agent.clone({ name: 'Clone' });
  const runner = new TemporalOpenAIRunner({ activityOptions: { startToCloseTimeout: '17s', summary: 'runner' } });
  const fallback = new TemporalModel('fallback-model', {
    activityOptions: { startToCloseTimeout: '90s', summary: 'fallback' },
  });
  const result = await runner.run(agent, 'Hello', { context: { user: 'test' }, runConfig: { model: fallback } });
  await runner.run(clone, 'Hello');
  await runner.run(agent, 'Hello', {
    context: { user: 'test' },
    activityOptions: { startToCloseTimeout: '40s', heartbeatTimeout: '2s', summary: 'run' },
  });
  await runner.run(new Agent({ name: 'Implicit' }), 'Hello', { runConfig: { model: fallback } });
  await runner.run(new Agent({ name: 'String', model: 'string-model' }), 'Hello', { runConfig: { model: fallback } });
  await runner.run(new Agent({ name: 'Default' }), 'Hello');
  return result.lastAgent === agent && agent.model === model && clone.model === model;
}

export async function temporalModelConcurrentWorkflow(): Promise<string[]> {
  const model = new TemporalModel('', {
    activityOptions: { startToCloseTimeout: '30s' },
    summaryProvider: { provide: (agent) => `model:${agent?.name}` },
  });
  const runner = new TemporalOpenAIRunner();
  return Promise.all(
    [11, 22].map(async (seconds) => {
      const agent = new Agent({
        name: `Agent${seconds}`,
        ...(seconds === 11 ? { model } : {}),
        instructions: async () => {
          await sleep('1ms');
          return `Instructions${seconds}`;
        },
      });
      const result = await runner.run(agent, 'Hello', {
        activityOptions: { startToCloseTimeout: `${seconds}s` },
        runConfig: { model },
      });
      return result.finalOutput ?? '';
    })
  );
}

export async function temporalModelNestedWorkflow(
  kind: 'handoff' | 'tool' | 'override' | 'fresh' | 'fallback-tool'
): Promise<string> {
  const child = new Agent({
    name: 'Child',
    ...(kind === 'fallback-tool'
      ? {}
      : {
          model: new TemporalModel('child', {
            activityOptions: { startToCloseTimeout: '33s' },
            summaryProvider: { provide: (agent) => agent!.name },
          }),
        }),
  });
  const parent = new Agent({
    name: 'Parent',
    model: new TemporalModel('parent', {
      activityOptions: { startToCloseTimeout: '60s' },
      summaryProvider: { provide: (agent) => agent!.name },
    }),
    ...(kind === 'handoff'
      ? { handoffs: [child] }
      : {
          tools: [
            kind === 'fresh'
              ? tool({
                  name: 'delegate',
                  description: 'Delegate',
                  parameters: {
                    type: 'object',
                    properties: { input: { type: 'string' } },
                    required: ['input'],
                    additionalProperties: false,
                  },
                  execute: async () => (await new TemporalOpenAIRunner().run(child, 'Hello')).finalOutput ?? '',
                })
              : child.asTool({
                  toolName: 'delegate',
                  runConfig: {
                    model: new TemporalModel('nested-fallback', { activityOptions: { startToCloseTimeout: '90s' } }),
                  },
                }),
          ],
        }),
  });
  const result = await new TemporalOpenAIRunner().run(
    parent,
    'Delegate',
    kind === 'override' || kind === 'fresh' ? { activityOptions: { startToCloseTimeout: '44s' } } : undefined
  );
  return result.finalOutput ?? '';
}

export async function temporalModelLocalWorkflow(): Promise<string> {
  const runner = new TemporalOpenAIRunner();
  const model = new TemporalModel('local', { useLocalActivity: true });
  const result = await runner.run(new Agent({ name: 'Local', model }), 'Hello');
  await runner.run(new Agent({ name: 'Run local' }), 'Hello', { useLocalActivity: true });
  return result.finalOutput ?? '';
}

export async function temporalModelValidationWorkflow(): Promise<string[]> {
  const errors: string[] = [];
  const runner = new TemporalOpenAIRunner();
  for (const options of [
    { activityOptions: { heartbeatTimeout: '1s' }, useLocalActivity: true },
    { activityOptions: { taskQueue: 'other' }, useLocalActivity: true },
    { activityOptions: { summary: 'static' }, summaryProvider: { provide: () => 'dynamic' } },
  ] satisfies TemporalModelOptions[]) {
    try {
      await runner.run(new Agent({ name: 'Invalid', model: new TemporalModel('test', options) }), 'Hello');
    } catch (err) {
      errors.push(err instanceof ApplicationFailure ? String(err.type) : String(err));
    }
  }
  for (const options of [{}, { streamingTopic: 'events', useLocalActivity: true }] satisfies TemporalModelOptions[]) {
    try {
      await runner.run(new Agent({ name: 'Invalid stream', model: new TemporalModel('test', options) }), 'Hello', {
        stream: true,
      });
    } catch (err) {
      errors.push(err instanceof ApplicationFailure ? String(err.type) : String(err));
    }
  }
  return errors;
}

export async function temporalModelStreamingWorkflow(kind: 'model' | 'run' | 'fallback'): Promise<string> {
  new WorkflowStream();
  const runner = new TemporalOpenAIRunner();
  const options: TemporalModelOptions = {
    activityOptions: { startToCloseTimeout: '21s' },
    streamingTopic: 'events',
    streamingBatchInterval: 123,
    useLocalActivity: false,
  };
  const agent = new Agent({
    name: 'Streaming',
    ...(kind !== 'fallback' ? { model: new TemporalModel('stream', kind === 'model' ? options : {}) } : {}),
  });
  const result = await runner.run(agent, 'Hello', {
    stream: true,
    ...(kind === 'run' ? { ...options } : {}),
    ...(kind === 'fallback' ? { runConfig: { model: new TemporalModel('fallback-stream', options) } } : {}),
  });
  for await (const _event of result) {
    /* Events are replayed from the Activity result. */
  }
  return result.finalOutput ?? '';
}

export async function temporalModelApprovalWorkflow(): Promise<string> {
  const confirm = tool({
    name: 'confirm',
    description: 'Confirm the action.',
    parameters: { type: 'object', properties: {}, required: [], additionalProperties: false },
    needsApproval: true,
    execute: async () => 'confirmed',
  });
  const agent = new Agent({
    name: 'Approval',
    model: new TemporalModel('approval', {
      activityOptions: { startToCloseTimeout: '33s' },
      summaryProvider: { provide: (agent) => agent!.name },
    }),
    tools: [confirm],
  });
  const runner = new TemporalOpenAIRunner();
  const first = await runner.run(agent, 'Confirm');
  const state = await RunState.fromString(agent, first.state.toString());
  for (const interruption of state.getInterruptions()) state.approve(interruption);
  const resumed = await runner.run(agent, state, { activityOptions: { startToCloseTimeout: '44s' } });
  return resumed.finalOutput ?? '';
}
