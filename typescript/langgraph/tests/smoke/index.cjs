const assert = require('node:assert/strict');
const { TestWorkflowEnvironment } = require('@temporalio/testing');
const { bundleWorkflowCode, DefaultLogger, Runtime, Worker } = require('@temporalio/worker');
const { LangGraphPlugin } = require('@temporalio/langgraph');
const { langGraph } = require('./graphs.cjs');

async function main() {
  Runtime.install({ logger: new DefaultLogger('WARN') });
  const plugin = new LangGraphPlugin(langGraph);
  const bundle = await bundleWorkflowCode({
    workflowsPath: require.resolve('./workflows.cjs'),
    plugins: [plugin],
    logger: new DefaultLogger('WARN'),
  });
  const env = await TestWorkflowEnvironment.createLocal();
  try {
    const taskQueue = 'langgraph-package-smoke';
    // No cached Workflows: every Workflow task replays the history from the start.
    const worker = await Worker.create({
      connection: env.nativeConnection,
      workflowBundle: bundle,
      taskQueue,
      plugins: [plugin],
      maxCachedWorkflows: 0,
    });
    await worker.runUntil(async () => {
      const options = { taskQueue, workflowExecutionTimeout: '30 seconds' };
      const graph = await env.client.workflow.execute('graphWorkflow', {
        ...options,
        workflowId: 'langgraph-package-smoke-graph',
        args: ['smoke'],
      });
      assert.equal(graph, 'smoke activity workflow');
      const functional = await env.client.workflow.execute('entrypointWorkflow', {
        ...options,
        workflowId: 'langgraph-package-smoke-entrypoint',
        args: ['smoke'],
      });
      assert.equal(functional, 'SMOKE');
    });
  } finally {
    await env.teardown();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
