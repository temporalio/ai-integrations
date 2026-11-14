/**
 * Packed-consumer regression test for a strict package layout.
 *
 * ADK 2.1 through 2.3 import `uuid` without declaring it. Under a hoisting layout
 * (npm, pnpm's default) the plugin's own `uuid` dependency happens to be
 * reachable from ADK's directory, so the shared clean-consumer smoke
 * (`typescript/_shared/smoke.mjs`, an npm install) cannot see the problem.
 * Under pnpm's isolated layout with `hoist=false` nothing undeclared is
 * reachable from any package, and `require('@temporalio/google-adk')` failed
 * with `Cannot find module 'uuid'` from inside ADK until the Worker-side
 * resolution hook (`src/adk-uuid-resolution.ts`) answered the request.
 *
 * The consumer here is that layout, built from the plugin's own synced
 * `node_modules` rather than by a package manager, so the test needs no
 * registry access (pnpm cannot resolve even an exact version offline without
 * cached metadata) and holds exactly the versions the lockfile pins: a
 * `.pnpm` virtual store in which the packed tarball sits beside links to its
 * declared dependencies, and a *copy* of the installed `@google/adk` sits
 * beside links to its own, with no hoisted fallback directory. The copy is
 * what makes the layout strict: Node resolves from a module's real path, and
 * from ADK's real path in the plugin's development install the walk up reaches
 * the checkout's `node_modules`, where `uuid` is the plugin's own direct
 * dependency (pnpm may hoist a copy into `node_modules/.pnpm/node_modules` too).
 */

import { spawnSync } from 'node:child_process';
import {
  constants as fsConstants,
  cpSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  renameSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import path from 'node:path';

import test from 'ava';

/** The plugin checkout (`__dirname` is `lib/__tests__` at run time). */
const pluginDir = path.resolve(__dirname, '..', '..');

const PLUGIN_PACKAGE = '@temporalio/google-adk';
const ADK_PACKAGE = '@google/adk';

interface Manifest {
  name: string;
  version: string;
  dependencies?: Record<string, string>;
  peerDependencies?: Record<string, string>;
  peerDependenciesMeta?: Record<string, { optional?: boolean }>;
}

function readManifest(dir: string): Manifest {
  return JSON.parse(readFileSync(path.join(dir, 'package.json'), 'utf8')) as Manifest;
}

/** Packs the plugin as CI does (`pnpm pack`), or with `npm pack` where pnpm is not on the path. */
function packPlugin(destination: string): string {
  mkdirSync(destination, { recursive: true });
  // The package managers are `.cmd` shims on Windows, which need a shell; a
  // shell gets the arguments joined, so a destination with a space is quoted.
  const shell = process.platform === 'win32';
  const failures: string[] = [];
  for (const [command, args] of [
    ['pnpm', ['pack', '--pack-destination', destination]],
    ['npm', ['pack', '--ignore-scripts', '--pack-destination', destination]],
  ] as const) {
    const quoted = shell ? args.map((arg) => (/\s/.test(arg) ? `"${arg}"` : arg)) : [...args];
    const result = spawnSync(command, quoted, { cwd: pluginDir, encoding: 'utf8', shell });
    if (result.status === 0) {
      const tarballs = readdirSync(destination).filter((name) => name.endsWith('.tgz'));
      if (tarballs.length !== 1) throw new Error(`${command} pack left ${tarballs.length} tarballs in ${destination}`);
      return path.join(destination, tarballs[0]!);
    }
    failures.push(`${command} pack: ${result.error?.message ?? result.stderr}`);
  }
  throw new Error(`could not pack the plugin:\n${failures.join('\n')}`);
}

/**
 * Extracts the tarball's `package/` directory under `destination` (a sibling of
 * the tarball's directory) and returns it. Both paths are passed relative to
 * their common parent, with forward slashes: a GNU tar on the path, as on the
 * Windows runners, reads an absolute `C:\\...` as a remote host.
 */
function extract(tarball: string, destination: string): string {
  mkdirSync(destination, { recursive: true });
  const cwd = path.dirname(destination);
  const relative = (target: string): string => path.relative(cwd, target).split(path.sep).join('/');
  const result = spawnSync('tar', ['-xzf', relative(tarball), '-C', relative(destination)], { cwd, encoding: 'utf8' });
  if (result.status !== 0) throw new Error(`tar failed: ${result.error?.message ?? result.stderr}`);
  return path.join(destination, 'package');
}

/** Every link the layout holds, most of them into the plugin checkout's own `node_modules`. */
const links: string[] = [];

/** A directory symlink; a junction on Windows, which needs no privilege. */
function linkDirectory(target: string, linkPath: string): void {
  mkdirSync(path.dirname(linkPath), { recursive: true });
  symlinkSync(target, linkPath, process.platform === 'win32' ? 'junction' : 'dir');
  links.push(linkPath);
}

/** Whether the installed ADK's CommonJS build requires `uuid` (2.1 through 2.3 do; 2.0 has no such import). */
function adkRequiresUuid(adk: string): boolean {
  const cjs = path.join(adk, 'dist', 'cjs');
  return readdirSync(cjs, { recursive: true, encoding: 'utf8' })
    .filter((entry) => entry.endsWith('.js'))
    .some((entry) => /require\(["']uuid["']\)/.test(readFileSync(path.join(cjs, entry), 'utf8')));
}

/**
 * The packages pnpm installed beside `packageDir` in its virtual store
 * (`<store>/<id>/node_modules/*`): the package's declared dependencies and
 * resolved peers, as `[name, real path]`, the package itself excluded.
 */
function storeSiblings(packageDir: string, name: string): Array<[string, string]> {
  const home = path.resolve(packageDir, ...name.split('/').map(() => '..'));
  const siblings: Array<[string, string]> = [];
  for (const entry of readdirSync(home)) {
    if (entry.startsWith('.')) continue;
    const names = entry.startsWith('@')
      ? readdirSync(path.join(home, entry)).map((inner) => `${entry}/${inner}`)
      : [entry];
    for (const sibling of names) {
      if (sibling !== name) siblings.push([sibling, realpathSync(path.join(home, sibling))]);
    }
  }
  return siblings;
}

interface StrictLayout {
  /** The consumer project root. */
  root: string;
  /** The packed plugin's real directory inside the consumer's virtual store. */
  plugin: string;
  /** The copied ADK's real directory inside the consumer's virtual store. */
  adk: string;
  adkManifest: Manifest;
}

/**
 * Builds pnpm's isolated `hoist=false` layout for a consumer that depends on
 * the packed plugin (`packedPlugin`, moved into place) and on the installed
 * `@google/adk`, from the plugin's own synced `node_modules`.
 */
function buildStrictLayout(root: string, packedPlugin: string): StrictLayout {
  const store = path.join(root, 'node_modules', '.pnpm');
  const manifest = readManifest(packedPlugin);

  // ADK: a copy, so that its real path (which Node resolves requests from) is
  // inside the consumer, with its own store siblings beside it and nothing else.
  const adkSource = realpathSync(path.join(pluginDir, 'node_modules', ADK_PACKAGE));
  const adkManifest = readManifest(adkSource);
  const adkHome = path.join(store, `${ADK_PACKAGE.replace('/', '+')}@${adkManifest.version}`, 'node_modules');
  const adk = path.join(adkHome, ADK_PACKAGE);
  cpSync(adkSource, adk, { recursive: true, mode: fsConstants.COPYFILE_FICLONE });
  for (const [name, target] of storeSiblings(adkSource, ADK_PACKAGE)) linkDirectory(target, path.join(adkHome, name));

  // The plugin: the packed tarball, with its declared dependencies beside it
  // (the plugin's own installed copies) and its required peers resolved as the
  // consumer resolves them, ADK being the copy above.
  const pluginHome = path.join(store, `${manifest.name.replace('/', '+')}@${manifest.version}`, 'node_modules');
  const plugin = path.join(pluginHome, manifest.name);
  mkdirSync(path.dirname(plugin), { recursive: true });
  renameSync(packedPlugin, plugin);
  for (const name of Object.keys({ ...manifest.dependencies, ...manifest.peerDependencies })) {
    if (manifest.peerDependenciesMeta?.[name]?.optional) continue;
    const target = name === ADK_PACKAGE ? adk : realpathSync(path.join(pluginDir, 'node_modules', name));
    linkDirectory(target, path.join(pluginHome, name));
  }

  // The consumer's own `node_modules`: its two direct dependencies, and no
  // hoisted fallback (`node_modules/.pnpm/node_modules`).
  linkDirectory(plugin, path.join(root, 'node_modules', manifest.name));
  linkDirectory(adk, path.join(root, 'node_modules', ADK_PACKAGE));
  writeFileSync(path.join(root, 'package.json'), JSON.stringify({ name: 'consumer', private: true }));
  return { root, plugin, adk, adkManifest };
}

/**
 * The probe the consumer runs in a fresh Node process: loads the plugin entry
 * named by `argv[2]` first, then ADK, and reports what each resolution
 * produced to the JSON file named by `argv[3]` (ADK writes to the console at
 * load, so stdout is not the channel).
 */
const PROBE_SOURCE = `
const { createRequire } = require('node:module');
const { writeFileSync } = require('node:fs');
const [entry, output] = process.argv.slice(2);
const report = {};
try {
  const loaded = require(entry);
  report.exports = Object.keys(loaded).sort();
  report.adkLoaded = typeof require('@google/adk').LlmAgent === 'function';
  report.adkUuid = createRequire(require.resolve('@google/adk')).resolve('uuid');
  report.pluginUuid = createRequire(require.resolve('@temporalio/google-adk')).resolve('uuid');
  if (typeof loaded.GoogleAdkPlugin === 'function') {
    const { webpackConfigHook } = new loaded.GoogleAdkPlugin().configureBundler({ workflowsPath: 'wf' });
    report.aliases = webpackConfigHook({ plugins: [] }).resolve.alias;
  }
} catch (err) {
  report.error = { code: err.code, message: err.message };
}
writeFileSync(output, JSON.stringify(report));
`;

/**
 * The same probe as an ES module of the consumer: Node's ESM loader hands the
 * plugin's CommonJS entry to its CommonJS loader, whose resolutions the hook
 * sees, so the plugin loads, and ADK's own ES module build, imported directly
 * afterwards, is resolved by the ESM loader alone, which the hook does not
 * reach.
 */
const ESM_PROBE_SOURCE = `
import { createRequire } from 'node:module';
import { writeFileSync } from 'node:fs';
const [entry, output] = process.argv.slice(2);
const report = {};
try {
  const loaded = await import(entry);
  report.exports = Object.keys(loaded).sort();
  const require = createRequire(import.meta.url);
  report.adkLoaded = typeof require('@google/adk').LlmAgent === 'function';
  report.adkUuid = createRequire(require.resolve('@google/adk')).resolve('uuid');
  report.pluginUuid = createRequire(require.resolve('@temporalio/google-adk')).resolve('uuid');
  try {
    await import('@google/adk');
    report.adkAsEsm = 'loaded';
  } catch (err) {
    report.adkAsEsm = err.code;
  }
} catch (err) {
  report.error = { code: err.code, message: err.message };
}
writeFileSync(output, JSON.stringify(report));
`;

interface ProbeReport {
  exports?: string[];
  adkLoaded?: boolean;
  adkUuid?: string;
  pluginUuid?: string;
  aliases?: Record<string, string>;
  adkAsEsm?: string;
  error?: { code?: string; message: string };
}

/**
 * The consumer's environment: the test's own, less `NODE_PATH`. The command
 * shims pnpm writes to `node_modules/.bin` (the `ava` one included) export
 * the hoisted virtual store (`node_modules/.pnpm/node_modules`) there, which
 * may hold a `uuid` of its own; without it the probes depend on the layout
 * alone.
 */
function consumerEnv(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  delete env.NODE_PATH;
  return env;
}

function probe(layout: StrictLayout, entry: string, script = 'probe.cjs'): ProbeReport {
  const output = path.join(layout.root, `${entry.replace(/[^a-z]+/gi, '-')}-${script.replace('.', '-')}.json`);
  const result = spawnSync(process.execPath, [path.join(layout.root, script), entry, output], {
    cwd: layout.root,
    env: consumerEnv(),
    encoding: 'utf8',
  });
  if (result.status !== 0) throw new Error(`the probe for ${entry} exited with ${result.status}:\n${result.stderr}`);
  return JSON.parse(readFileSync(output, 'utf8')) as ProbeReport;
}

let workspace: string | undefined;
let layout: StrictLayout;

test.before(() => {
  // The real path: paths the consumer reports are real paths too (macOS keeps
  // its temporary directory behind a symlink).
  workspace = realpathSync(mkdtempSync(path.join(tmpdir(), 'google-adk-strict-')));
  const tarball = packPlugin(path.join(workspace, 'pack'));
  const extracted = extract(tarball, path.join(workspace, 'extract'));
  layout = buildStrictLayout(path.join(workspace, 'consumer'), extracted);
  writeFileSync(path.join(layout.root, 'probe.cjs'), PROBE_SOURCE);
  writeFileSync(path.join(layout.root, 'probe.mjs'), ESM_PROBE_SOURCE);
});

test.after.always(() => {
  // The links lead into the plugin checkout: each goes as a link first, so
  // that the removal of the workspace has none left to follow.
  for (const link of links) rmSync(link, { force: true });
  if (workspace) rmSync(workspace, { recursive: true, force: true });
});

/** Runs `source` in a fresh Node process in the consumer, as the consumer. */
function runInConsumer(source: string): ReturnType<typeof spawnSync> {
  return spawnSync(process.execPath, ['-e', source], { cwd: layout.root, env: consumerEnv(), encoding: 'utf8' });
}

// The layout is strict: `uuid` is unreachable from ADK's directory, so ADK alone cannot load (E2E)
test('strictLayoutLeavesAdkWithoutUuid', (t) => {
  if (layout.adkManifest.dependencies?.uuid !== undefined) {
    t.log(`${ADK_PACKAGE}@${layout.adkManifest.version} declares uuid; nothing is left for the hook to answer`);
    t.pass();
    return;
  }
  const resolution = runInConsumer(
    `try { require('node:module').createRequire(${JSON.stringify(path.join(layout.adk, 'package.json'))}).resolve('uuid'); ` +
      `console.log('resolved'); } catch (err) { console.log(err.code); }`
  );
  t.is(resolution.status, 0, String(resolution.stderr));
  t.is(String(resolution.stdout).trim(), 'MODULE_NOT_FOUND');

  const load = runInConsumer(`require('${ADK_PACKAGE}')`);
  if (adkRequiresUuid(layout.adk)) {
    t.not(load.status, 0);
    t.regex(String(load.stderr), /Cannot find module 'uuid'/);
  } else {
    t.log(`${ADK_PACKAGE}@${layout.adkManifest.version} does not import uuid; nothing is left for the hook to answer`);
    t.is(load.status, 0, String(load.stderr));
  }
});

// Each plugin entry, loaded first, answers ADK's `uuid` import with the plugin's own copy (E2E)
for (const [label, entry, exported] of [
  ['Main', PLUGIN_PACKAGE, 'GoogleAdkPlugin'],
  ['Workflow', `${PLUGIN_PACKAGE}/workflow`, 'TemporalModel'],
  ['Testing', `${PLUGIN_PACKAGE}/testing`, 'fakeModelProvider'],
] as const) {
  test(`strictLayoutConsumerLoads${label}Entry`, (t) => {
    const report = probe(layout, entry);
    t.is(report.error, undefined, report.error && `${report.error.code}: ${report.error.message}`);
    t.true(report.exports?.includes(exported));
    t.true(report.adkLoaded);
    const pluginUuid = createRequire(path.join(layout.plugin, 'package.json')).resolve('uuid');
    t.is(report.pluginUuid, pluginUuid);
    t.is(report.adkUuid, pluginUuid);
    if (label === 'Main') {
      // The bundler pins resolve inside the consumer too: ADK's per-file common
      // surface from the consumer's ADK, not from the plugin checkout.
      t.is(report.aliases?.[`${ADK_PACKAGE}$`], path.join(layout.adk, 'dist', 'esm', 'common.js'));
    }
  });
}

// An ES module of the consumer loads the plugin the same way; ADK's own ES module build stays out of reach (E2E)
test('strictLayoutEsmConsumerLoadsTheMainEntry', (t) => {
  const report = probe(layout, PLUGIN_PACKAGE, 'probe.mjs');
  t.is(report.error, undefined, report.error && `${report.error.code}: ${report.error.message}`);
  t.true(report.exports?.includes('GoogleAdkPlugin'));
  t.true(report.adkLoaded);
  const pluginUuid = createRequire(path.join(layout.plugin, 'package.json')).resolve('uuid');
  t.is(report.pluginUuid, pluginUuid);
  t.is(report.adkUuid, pluginUuid);
  const adkImportsUuid = layout.adkManifest.dependencies?.uuid === undefined && adkRequiresUuid(layout.adk);
  t.is(report.adkAsEsm, adkImportsUuid ? 'ERR_MODULE_NOT_FOUND' : 'loaded');
});
