// The pinned upstream monorepo does not ship dist files or build Git dependencies.
// Build its two packages using its own locked tooling, without changing our lock.
import { execFileSync } from 'node:child_process';
import { cp, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const pluginDir = fileURLToPath(new URL('..', import.meta.url));
const manifest = JSON.parse(await readFile(join(pluginDir, 'package.json'), 'utf8'));
const packages = ['agents-core', 'agents-openai'];
const commits = packages.map((name) => {
  const source = manifest.devDependencies[`@openai/${name}`];
  const match = /^github:openai\/openai-agents-js#([a-f0-9]{40})&path:\/packages\/(agents-core|agents-openai)$/.exec(source);
  if (!match || match[2] !== name) throw new Error(`Expected a commit-pinned source for @openai/${name}`);
  return match[1];
});
const commit = commits[0];
if (commits.some((value) => value !== commit)) throw new Error('OpenAI Agents packages must use the same commit');

const destination = (name) => join(pluginDir, 'node_modules', '@openai', name, 'dist');
const built = await Promise.all(
  packages.map(async (name) => {
    try {
      const marker = await readFile(join(destination(name), '.temporal-upstream-commit'), 'utf8');
      await Promise.all(['index.js', 'index.mjs', 'index.d.ts'].map((file) => readFile(join(destination(name), file))));
      return marker === commit;
    } catch {
      return false;
    }
  }),
);
if (built.every(Boolean)) process.exit(0);

const checkout = await mkdtemp(join(tmpdir(), 'temporal-openai-agents-'));
function run(command, args) {
  execFileSync(command, args, { cwd: checkout, stdio: 'inherit', shell: process.platform === 'win32' });
}

try {
  run('git', ['init', '--quiet']);
  run('git', ['fetch', '--depth=1', 'https://github.com/openai/openai-agents-js.git', commit]);
  run('git', ['checkout', '--quiet', '--detach', 'FETCH_HEAD']);

  // Keep the upstream lock but omit its examples, docs, and unrelated packages.
  const workspacePath = join(checkout, 'pnpm-workspace.yaml');
  const workspace = (await readFile(workspacePath, 'utf8')).replaceAll('\r\n', '\n');
  await writeFile(
    workspacePath,
    workspace.replace(
      /packages:\n(?:  - .+\n)+/,
      `packages:\n${packages.map((name) => `  - packages/${name}\n`).join('')}`,
    ),
  );
  const buildConfigPath = join(checkout, 'tsc-multi.json');
  const buildConfig = JSON.parse(await readFile(buildConfigPath, 'utf8'));
  buildConfig.projects = packages.map((name) => `packages/${name}/tsconfig.json`);
  await writeFile(buildConfigPath, JSON.stringify(buildConfig));

  const upstream = JSON.parse(await readFile(join(checkout, 'package.json'), 'utf8'));
  const packageManager = /^pnpm@(\d+\.\d+\.\d+)(?:\+.*)?$/.exec(upstream.packageManager);
  if (!packageManager) throw new Error('Expected an upstream pnpm version');
  const pnpm = (args) => run('npx', ['--yes', `pnpm@${packageManager[1]}`, ...args]);
  pnpm(['install', '--frozen-lockfile', '--ignore-scripts']);
  // tsx also supports Node versions before native TypeScript execution was enabled.
  pnpm(['--filter', '@openai/agents-core', '--filter', '@openai/agents-openai', 'exec', 'tsx', '../../scripts/embedMeta.mts']);
  pnpm(['exec', 'tsc-multi', '--maxWorkers', '1']);

  for (const name of packages) {
    await rm(destination(name), { recursive: true, force: true });
    await cp(join(checkout, 'packages', name, 'dist'), destination(name), { recursive: true });
    await writeFile(join(destination(name), '.temporal-upstream-commit'), commit);
  }
} finally {
  await rm(checkout, { recursive: true, force: true });
}
