/** Run the plugin's independently locked ESLint with declared shared configuration. */
import { createRequire } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
const requirePlugin=createRequire(resolve('package.json'));
const binary=join(dirname(requirePlugin.resolve('eslint/package.json')),'bin/eslint.js');
const result=spawnSync(process.env.JS_BINARY__NODE_BINARY || process.execPath,[binary,'--config','../_shared/eslint.config.mjs',...process.argv.slice(2)],{stdio:'inherit',env:process.env});
if(result.error)throw result.error;
process.exitCode=result.status ?? 1;
