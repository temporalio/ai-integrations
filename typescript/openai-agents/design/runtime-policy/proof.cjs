const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { TestWorkflowEnvironment } = require('@temporalio/testing');
const { Worker, bundleWorkflowCode, Runtime, DefaultLogger } = require('@temporalio/worker');
const { OpenAIAgentsPlugin } = require('./lib');
const { textResponse, toolCallResponse, handoffResponse } = require('./lib/__tests__/stubs/openai-agents-fakes');
require('@openai/agents-core').setTracingDisabled(true);

Runtime.install({ logger: new DefaultLogger('ERROR') });
const mode = process.argv[2];
const outputDirectory = process.argv[3];
const cases = ['copy-listener', 'copy-handoff', 'same-listener', 'same-handoff',
  'schema-provider', 'undefined-provider', 'input-model', 'tool-provider', 'copy-native', 'copy-string', 'native',
  'input-guard', 'output-guard', 'approval'];
const inline = [];
const requests = [];
const provider = { getModel(name) { return {
  async getResponse(request) {
    requests.push(name);
    if (request.systemInstructions === 'Route.') return handoffResponse('transfer_to_desk');
    if (request.systemInstructions === 'Use the tool.') return toolCallResponse('ask_nested', { input: 'q' });
    return textResponse('nested answer');
  },
}; } };
const plugin = new OpenAIAgentsPlugin({ modelProvider: provider });
const { requireRuntimeHooks } = require('./boundary-require-runtime-hooks.cjs');
const bundlerPlugins = mode === 'policy' ? [plugin, requireRuntimeHooks] : [plugin];
const sinks = { report: { inline: { fn(_info, kind) { inline.push(kind); }, callDuringReplay: true } } };

(async () => {
  fs.mkdirSync(outputDirectory, { recursive: true });
  console.log(JSON.stringify({ mode, node: process.version,
    core: JSON.parse(fs.readFileSync(path.join(path.dirname(require.resolve('@openai/agents-core')), '../package.json'))).version }));
  const env = await TestWorkflowEnvironment.createLocal({
    server: { executable: { type: 'existing-path', path: '/usr/local/bin/temporal' } },
  });
  try {
    const workflowBundle = await bundleWorkflowCode({ workflowsPath: require.resolve('./boundary-workflows.cjs'),
      plugins: bundlerPlugins, logger: new DefaultLogger('ERROR') });
    fs.writeFileSync(path.join('.boundary-originals', `${mode}-bundle.js`), workflowBundle.code);
    const worker = await Worker.create({ connection: env.nativeConnection, workflowBundle,
      taskQueue: `boundary-${mode}`, maxCachedWorkflows: 0, plugins: [plugin], sinks });
    await worker.runUntil(async () => {
      for (const kind of cases) {
        const before = inline.length;
        const beforeRequests = requests.length;
        const handle = await env.client.workflow.start('boundaryProof', { taskQueue: `boundary-${mode}`,
          workflowId: `${mode}-${kind}`, args: [kind], workflowExecutionTimeout: '30s' });
        const result = await handle.result();
        const history = await handle.fetchHistory();
        fs.writeFileSync(path.join(outputDirectory, `${kind}.json`), JSON.stringify(history.toJSON()));
        const liveInline = inline.length - before;
        await Worker.runReplayHistory({ workflowBundle, plugins: [plugin], sinks }, history);
        const replayInline = inline.length - before - liveInline;
        const activities = history.events.filter(event => event.activityTaskScheduledEventAttributes)
          .map(event => event.activityTaskScheduledEventAttributes.activityType.name);
        const row = { kind, ...result, liveInline, replayInline, activities,
          modelRequests: requests.slice(beforeRequests) };
        console.log(JSON.stringify(row));
        assert.equal(result.separateCopies, true);
        if (mode === 'policy') {
          assert.equal(result.patched, false);
          assert.equal(liveInline, 0);
          assert.equal(replayInline, 0);
          if (kind === 'native' || kind === 'copy-string') {
            assert.equal(result.output, 'nested answer');
            assert.equal(activities.length, kind === 'native' ? 2 : 1);
          } else if (kind === 'input-guard' || kind === 'output-guard') {
            assert.equal(result.output, 'BLOCKED');
            assert.equal(result.guardCalls, 1);
            assert.equal(activities.length, kind === 'input-guard' ? 1 : 2);
          } else if (kind === 'approval') {
            assert.equal(result.interruptions, 1);
            assert.equal(activities.length, 1);
          } else assert.match(result.output, /Runtime policy/);
        } else if (kind === 'copy-listener' || kind === 'copy-handoff') {
          assert.equal(result.output, 'INLINE');
          assert.equal(liveInline, 1);
          assert.equal(replayInline, 1);
        } else {
          assert.equal(liveInline, 0);
          assert.equal(replayInline, 0);
          if (kind === 'native' || kind === 'copy-string') assert.equal(result.output, 'nested answer');
          else assert.notEqual(result.output, 'INLINE');
        }
        if (kind === 'input-guard' || kind === 'output-guard') {
          assert.equal(result.output, 'BLOCKED'); assert.equal(result.guardCalls, 1);
        }
        if (kind === 'approval') assert.equal(result.interruptions, 1);
      }
    });
  } finally { await env.teardown(); }
  console.log(`PASS ${mode}: ${cases.length} cases, each replayed`);
  if (mode === 'policy') {
    const root = path.join(path.dirname(require.resolve('@openai/agents-core-copy')), '..');
    const manifest = path.join(root, 'package.json');
    const run = path.join(root, 'dist/run.js');
    const originalManifest = fs.readFileSync(manifest);
    const originalRun = fs.readFileSync(run);
    try {
      const legacy = JSON.parse(originalManifest);
      delete legacy.exports['./runtime-policy'];
      fs.writeFileSync(manifest, JSON.stringify(legacy));
      fs.copyFileSync('.boundary-originals/duplicate-run.js', run);
      const messages = [];
      const logger = new DefaultLogger('ERROR', entry => messages.push(entry.message));
      await assert.rejects(bundleWorkflowCode({ workflowsPath: require.resolve('./boundary-workflows.cjs'),
        plugins: bundlerPlugins, logger }), /Webpack finished with errors/);
      assert.match(messages.join('\n'), /Unsupported agents-core runtime: .*agents-core-copy/);
      console.log(messages.join('\n').split('\n').find(line => line.includes('Unsupported agents-core runtime')));
      console.log('PASS mixed-version bundle rejection: an unmodified SDK copy never reaches execution');
    } finally {
      fs.writeFileSync(manifest, originalManifest);
      fs.writeFileSync(run, originalRun);
    }
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
