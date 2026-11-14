/**
 * Unit coverage for the Worker-side `uuid` resolution hook
 * (`src/adk-uuid-resolution.ts`): a `uuid` request Node cannot resolve
 * resolves to the plugin's copy when it comes from inside `@google/adk`, every
 * other request is left to Node, the hook installs once per process, and it
 * stays out where the plugin's own `uuid` is missing. The strict-layout
 * consumer test covers the real thing end to end.
 *
 * The resolutions run in a child process without `NODE_PATH`: the command
 * shims pnpm writes to `node_modules/.bin` (the `ava` one included) export
 * the hoisted virtual store (`node_modules/.pnpm/node_modules`) there, and a
 * `uuid` hoisted into it would answer the requests below without the hook.
 */

import { spawnSync } from 'node:child_process';
import { copyFileSync, mkdtempSync, rmSync } from 'node:fs';
import Module from 'node:module';
import { tmpdir } from 'node:os';
import path from 'node:path';

import test from 'ava';

import { installAdkUuidResolution } from '../adk-uuid-resolution';

/**
 * Resolves three requests through `Module._resolveFilename` after installing
 * the hook, each from a requesting module with no `node_modules` lookup paths
 * of its own (the shape `resolve-from` and similar hand the resolver), so a
 * bare request fails unless the hook answers it.
 */
const PROBE_SOURCE = `
const Module = require('node:module');
const path = require('node:path');
require(process.env.ADK_UUID_HOOK);
const adkFile = path.join(path.sep, 'app', 'node_modules', '@google', 'adk', 'dist', 'cjs', 'plugins', 'reflect_retry_model_plugin.js');
const otherFile = path.join(path.sep, 'app', 'node_modules', 'other', 'index.js');
function resolve(request, filename) {
  try {
    return { resolved: Module._resolveFilename(request, { id: filename, filename, paths: [] }, false) };
  } catch (err) {
    return { code: err.code };
  }
}
process.stdout.write(JSON.stringify({
  installed: Module._resolveFilename[Symbol.for('@temporalio/google-adk.adkUuidResolution')] === true,
  uuidFromAdk: resolve('uuid', adkFile),
  uuidFromElsewhere: resolve('uuid', otherFile),
  otherFromAdk: resolve('@temporalio/no-such-package', adkFile),
}));
`;

interface Resolution {
  resolved?: string;
  code?: string;
}

interface ProbeReport {
  installed: boolean;
  uuidFromAdk: Resolution;
  uuidFromElsewhere: Resolution;
  otherFromAdk: Resolution;
}

function probe(hook = require.resolve('../adk-uuid-resolution')): ProbeReport {
  const env: NodeJS.ProcessEnv = { ...process.env, ADK_UUID_HOOK: hook };
  delete env.NODE_PATH;
  const result = spawnSync(process.execPath, ['-e', PROBE_SOURCE], { env, encoding: 'utf8' });
  if (result.status !== 0) throw new Error(`the resolution probe exited with ${result.status}:\n${result.stderr}`);
  return JSON.parse(result.stdout) as ProbeReport;
}

// ADK's `uuid` request, unresolvable on its own, resolves to the plugin's copy
test('uuidRequestFromAdkResolvesToThePluginCopy', (t) => {
  const report = probe();
  t.true(report.installed);
  t.deepEqual(report.uuidFromAdk, { resolved: require.resolve('uuid') });
});

// Without the plugin's own `uuid` to answer with, the hook leaves Node's resolver alone
test('hookWithoutItsOwnUuidLeavesTheResolverAlone', (t) => {
  // The compiled hook on its own in a directory with no `uuid` in reach.
  const dir = mkdtempSync(path.join(tmpdir(), 'google-adk-hook-'));
  t.teardown(() => rmSync(dir, { recursive: true, force: true }));
  const hook = path.join(dir, 'adk-uuid-resolution.js');
  copyFileSync(require.resolve('../adk-uuid-resolution'), hook);
  const report = probe(hook);
  t.false(report.installed);
  t.deepEqual(report.uuidFromAdk, { code: 'MODULE_NOT_FOUND' });
});

// The same request from any other package is left to Node, and fails as it would
test('uuidRequestFromElsewhereIsLeftToNode', (t) => {
  t.deepEqual(probe().uuidFromElsewhere, { code: 'MODULE_NOT_FOUND' });
});

// Only `uuid` is answered: another undeclared request from ADK still fails
test('otherRequestsFromAdkAreLeftToNode', (t) => {
  t.deepEqual(probe().otherFromAdk, { code: 'MODULE_NOT_FOUND' });
});

// Installing again (a second plugin copy, a re-import) leaves the installed wrapper alone
test('installsOncePerProcess', (t) => {
  const resolver = (): unknown => (Module as unknown as { _resolveFilename: unknown })._resolveFilename;
  const installed = resolver();
  installAdkUuidResolution();
  t.is(resolver(), installed);
});

// The test file itself is not what proves the hook: the module path it loads is
test('probeLoadsTheCompiledHook', (t) => {
  t.is(path.basename(require.resolve('../adk-uuid-resolution')), 'adk-uuid-resolution.js');
});
