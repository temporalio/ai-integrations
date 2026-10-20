/** Resolve the pinned dev server through runfiles before AVA changes directories. */
import { runfiles } from '@bazel/runfiles';
import { createRequire } from 'node:module';
import { resolve, dirname, join } from 'node:path';
import { spawnSync } from 'node:child_process';
// MCP intentionally supplies a minimal child environment. Use the real pinned Node
// executable so those children do not depend on a Bazel wrapper's environment.
const realNode = process.env.JS_BINARY__NODE_BINARY;
if(realNode) {
  process.execPath=realNode;
  process.env.PATH=dirname(realNode)+require('node:path').delimiter+(process.env.PATH || '');
}
const requirePlugin = createRequire(resolve('package.json'));
const path = process.env.TEMPORAL_CLI_RUNFILE;
if (path) process.env.TEMPORAL_CLI_PATH = runfiles.resolve(path);
const result = spawnSync(process.execPath, ['--require', require.resolve('./real-node.js'), join(dirname(requirePlugin.resolve('ava')), 'cli.mjs'), ...process.argv.slice(2)], { stdio: 'inherit', env: process.env });
if(result.error)throw result.error;
process.exitCode=result.status ?? 1;
