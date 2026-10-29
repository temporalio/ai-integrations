const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { TestWorkflowEnvironment } = require('@temporalio/testing');
const { Worker, Runtime, DefaultLogger } = require('@temporalio/worker');
const { temporal } = require('@temporalio/proto');
const { OpenAIAgentsPlugin } = require('./lib');
const { textResponse, toolCallResponse } = require('./lib/__tests__/stubs/openai-agents-fakes');
require('@openai/agents-core').setTracingDisabled(true);
Runtime.install({ logger: new DefaultLogger('ERROR') });
const evidence = process.argv[2];
const bundles = Object.fromEntries(['baseline', 'policy'].map(mode => [mode,
  { code: fs.readFileSync(path.join('.boundary-originals', `${mode}-bundle.js`), 'utf8') }]));
const provider = { getModel() { return { async getResponse(request) {
  return request.systemInstructions === 'Use the tool.'
    ? toolCallResponse('ask_nested', { input: 'q' }) : textResponse('nested answer');
} }; } };
const plugin = new OpenAIAgentsPlugin({ modelProvider: provider });
const options = mode => ({ workflowBundle: bundles[mode], plugins: [plugin] });
function history(mode, kind) {
  return temporal.api.history.v1.History.fromObject(JSON.parse(fs.readFileSync(
    path.join(evidence, `histories-${mode}`, `${kind}.json`))));
}
(async () => {
  for (const source of ['baseline', 'policy']) for (const target of ['baseline', 'policy']) {
    for (const kind of ['native', 'copy-string']) {
      await Worker.runReplayHistory(options(target), history(source, kind));
      console.log(`PASS finished replay ${source} -> ${target}: ${kind}`);
    }
  }
  const changed = history('policy', 'native');
  changed.events.find(event => event.activityTaskScheduledEventAttributes)
    .activityTaskScheduledEventAttributes.activityType.name = 'deliberatelyDifferentActivity';
  await assert.rejects(Worker.runReplayHistory(options('policy'), changed), /TMPRL1100/);
  console.log('PASS negative replay control: [TMPRL1100]');
  const env = await TestWorkflowEnvironment.createLocal({
    server: { executable: { type: 'existing-path', path: '/usr/local/bin/temporal' } },
  });
  try {
    for (const [source, target] of [['baseline', 'policy'], ['policy', 'baseline']]) {
      const taskQueue = `carry-${source}-${target}`;
      let handle;
      const first = await Worker.create({ ...options(source), connection: env.nativeConnection,
        taskQueue, maxCachedWorkflows: 0 });
      await first.runUntil(async () => {
        handle = await env.client.workflow.start('boundaryProof', { taskQueue,
          workflowId: taskQueue, args: ['native-held'], workflowExecutionTimeout: '30s' });
        for (let attempts = 0; attempts < 100; attempts++) {
          try { if (await handle.query('held')) return; } catch {}
          await new Promise(resolve => setTimeout(resolve, 50));
        }
        throw new Error('Workflow did not reach the held state');
      });
      const second = await Worker.create({ ...options(target), connection: env.nativeConnection,
        taskQueue, maxCachedWorkflows: 0 });
      await second.runUntil(async () => {
        await handle.signal('release');
        assert.equal((await handle.result()).output, 'nested answer');
      });
      const completed = await handle.fetchHistory();
      fs.writeFileSync(path.join(evidence, `carry-${source}-${target}.json`), JSON.stringify(completed.toJSON()));
      await Worker.runReplayHistory(options(target), completed);
      console.log(`PASS open run across Worker replacement ${source} -> ${target}`);
    }
  } finally { await env.teardown(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
