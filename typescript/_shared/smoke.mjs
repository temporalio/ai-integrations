// Install the tarball in a clean consumer with no access to plugin dev dependencies.
import { execFileSync } from 'node:child_process';
import { cpSync, existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const pkg = JSON.parse(readFileSync('package.json', 'utf8'));
const tarballs = readdirSync('dist').filter((name) => name.endsWith('.tgz'));
if (tarballs.length !== 1) throw new Error('Expected exactly one npm tarball in dist');
const tarball = resolve('dist', tarballs[0]);
const workflowSmoke = resolve('tests/smoke');
const registryVersion = process.env.NPM_REGISTRY_VERSION;
const dependencies = [registryVersion ? `${pkg.name}@${registryVersion}` : tarball];
if (existsSync(workflowSmoke)) {
  for (const dependency of ['@temporalio/worker', '@temporalio/testing']) {
    dependencies.push(`${dependency}@${pkg.devDependencies[dependency]}`);
  }
}
const consumer = mkdtempSync(join(tmpdir(), 'temporal-plugin-smoke-'));
try {
  writeFileSync(join(consumer, 'package.json'), JSON.stringify({ name: 'consumer', private: true }));
  execFileSync('npm', ['install', '--ignore-scripts', '--no-audit', '--no-fund', ...dependencies], {
    cwd: consumer,
    stdio: 'inherit',
    shell: process.platform === 'win32',
  });
  execFileSync(process.execPath, ['-e', `require(${JSON.stringify(pkg.name)})`], {
    cwd: consumer,
    stdio: 'inherit',
  });
  const installed = JSON.parse(readFileSync(join(consumer, 'node_modules', pkg.name, 'package.json'), 'utf8'));
  if (installed.version !== (registryVersion || pkg.version)) throw new Error('Installed version differs from release');
  console.log(`OK: clean consumer imports ${pkg.name}`);
  if (existsSync(workflowSmoke)) {
    cpSync(workflowSmoke, join(consumer, 'smoke'), { recursive: true });
    execFileSync(process.execPath, ['smoke/index.cjs'], {
      cwd: consumer,
      stdio: 'inherit',
    });
    console.log(`OK: clean consumer bundles and runs ${pkg.name} workflows`);
  }
} finally {
  rmSync(consumer, { recursive: true, force: true });
}
