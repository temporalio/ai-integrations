const assert = require('node:assert/strict');
const { Model } = require('@strands-agents/sdk');
const { TestWorkflowEnvironment } = require('@temporalio/testing');
const { bundleWorkflowCode, DefaultLogger, Runtime, Worker } = require('@temporalio/worker');
const { StrandsPlugin } = require('@temporalio/strands-agents');

class LocalModel extends Model {
  updateConfig() {}
  getConfig() {
    return {};
  }
  async *stream() {
    yield { type: 'modelMessageStartEvent', role: 'assistant' };
    yield { type: 'modelContentBlockStartEvent' };
    yield { type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text: 'smoke reply' } };
    yield { type: 'modelContentBlockStopEvent' };
    yield { type: 'modelMessageStopEvent', stopReason: 'endTurn' };
  }
}

async function main() {
  Runtime.install({ logger: new DefaultLogger('WARN') });
  const plugin = new StrandsPlugin({ models: { local: () => new LocalModel() } });
  const bundle = await bundleWorkflowCode({
    workflowsPath: require.resolve('./workflows.cjs'),
    plugins: [plugin],
    logger: new DefaultLogger('WARN'),
  });
  const env = await TestWorkflowEnvironment.createLocal({
    server: process.env.TEMPORAL_CLI_PATH
      ? { executable: { type: 'existing-path', path: process.env.TEMPORAL_CLI_PATH } }
      : undefined,
  });
  try {
    const taskQueue = 'strands-package-smoke';
    const worker = await Worker.create({
      connection: env.nativeConnection,
      workflowBundle: bundle,
      taskQueue,
      plugins: [plugin],
    });
    const result = await worker.runUntil(
      env.client.workflow.execute('agentWorkflow', {
        workflowId: 'strands-package-smoke',
        taskQueue,
        args: ['Hello'],
        workflowExecutionTimeout: '30 seconds',
      })
    );
    assert.equal(result, 'smoke reply');
  } finally {
    await env.teardown();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
