/** Deterministic npm archives and isolated installed-package verification. */
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve, relative, sep } from 'node:path';
import { createRequire } from 'node:module';
import { create } from 'tar';
import { runfiles } from '@bazel/runfiles';
import { archiveFiles } from './archive';
import { Document, argumentsOf, command, json, root, toml } from './common';

export function walk(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap(e => e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)]);
}
export function targets(value: any): string[] {
  if (typeof value === 'string') return [value];
  return value && typeof value === 'object' ? Object.values(value).flatMap(targets) : [];
}
export async function checkPackage(project: string, archive: string, version?: string): Promise<void> {
  const files = await archiveFiles(archive), manifest = json(join(project, 'package.json'));
  if (version) manifest.version = version;
  const packed = JSON.parse(files.get('package/package.json')?.toString() || '{}');
  for (const key of ['name', 'version', 'dependencies', 'peerDependencies', 'exports']) {
    if (JSON.stringify(packed[key]) !== JSON.stringify(manifest[key])) throw new Error(`packaged ${key} differs`);
  }
  if (!files.get('package/LICENSE')?.equals(readFileSync(join(root(), 'LICENSE')))) throw new Error('packaged LICENSE differs');
  for (const name of files.keys()) if (name.split('/').some(p => ['tests', '__tests__', 'node_modules'].includes(p))) throw new Error(`test/dependency file shipped: ${name}`);
  for (const target of targets([packed.main, packed.types, packed.exports])) if (!files.has(`package/${target.replace(/^\.\//, '')}`)) throw new Error(`missing entrypoint: ${target}`);
  for (const source of walk(join(project, 'src')).filter(p => p.endsWith('.ts') && !p.split('/').includes('__tests__'))) {
    const path = relative(project, source).split('\\').join('/');
    if (!files.get(`package/${path}`)?.equals(readFileSync(source))) throw new Error(`packaged source differs: ${path}`);
    if (!files.has(`package/lib/${relative(join(project, 'src'), source).replace(/\.ts$/, '.js').split('\\').join('/')}`)) throw new Error(`missing compiled source: ${path}`);
  }
}
export async function pack(project: string, compiled: string, output: string, version: string, declarations?: string): Promise<void> {
  const temp = mkdtempSync(join(tmpdir(), 'npm-pack-'));
  try {
    const staged = join(temp, 'package');
    cpSync(project, staged, { recursive: true, filter: p => !relative(project, p).split(/[\\/]/).some(s => ['node_modules','dist','lib','.git','__tests__','tests'].includes(s)) });
    cpSync(compiled, join(staged, 'lib'), { recursive: true, dereference: true, filter: p => !relative(compiled,p).split(/[\\/]/).includes('__tests__') && (!declarations || !p.endsWith('.d.ts')) });
    if (declarations) for (const file of walk(declarations)) {
      const target = join(staged, 'lib', relative(declarations, file));
      mkdirSync(dirname(target), { recursive: true });
      writeFileSync(target, readFileSync(file));
    }
    const manifest = json(join(staged, 'package.json')); manifest.version = version;
    writeFileSync(join(staged, 'package.json'), JSON.stringify(manifest, null, 2) + '\n');
    const included = walk(staged).filter(p => ['package.json','README.md','LICENSE'].includes(relative(staged,p)) || /^(src|lib)[\\/]/.test(relative(staged,p))).map(p => relative(temp,p));
    await create({ file: output, cwd: temp, gzip: true, portable: true, mtime: new Date('1980-01-01T00:00:00Z') }, included.sort());
  } finally { rmSync(temp, { recursive: true, force: true }); }
}
export async function smoke(project: string, archive: string): Promise<void> {
  if (process.env.TEMPORAL_CLI_RUNFILE) process.env.TEMPORAL_CLI_PATH = runfiles.resolve(process.env.TEMPORAL_CLI_RUNFILE);
  const temp = realpathSync(mkdtempSync(join(tmpdir(), 'npm-consumer-')));
  try {
    writeFileSync(join(temp, 'package.json'), '{"private":true}');
    const manifest = json(join(project, 'package.json'));
    const packedVersion=JSON.parse((await archiveFiles(archive)).get('package/package.json')!.toString()).version;
    const registryVersion=process.env.NPM_REGISTRY_VERSION;
    if(registryVersion && registryVersion!==packedVersion)throw new Error('registry version differs from the tested tarball');
    const npmCli = join(dirname(require.resolve('npm/package.json')), 'bin/npm-cli.js');
    command(process.env.JS_BINARY__NODE_BINARY || process.execPath, [npmCli, 'install','--ignore-scripts','--no-audit','--no-fund', registryVersion ? `${manifest.name}@${registryVersion}` : resolve(archive)], temp);
    const meta = toml(join(project, 'plugin.toml'));
    const workflowSmoke=join(project,'tests/smoke');
    if(existsSync(workflowSmoke)) {
      command(process.env.JS_BINARY__NODE_BINARY || process.execPath, [npmCli, 'install','--ignore-scripts','--no-audit','--no-fund', ...['@temporalio/worker','@temporalio/testing'].map(name=>`${name}@${manifest.devDependencies[name]}`)],temp);
    }
    const consumerRequire = createRequire(join(temp, 'package.json'));
    for (const name of meta.smoke.imports) {
      const loaded = consumerRequire.resolve(name);
      if (!realpathSync(loaded).startsWith(join(temp, 'node_modules') + sep)) throw new Error(`unexpected package origin: ${loaded}`);
      consumerRequire(name);
    }
    if(existsSync(workflowSmoke)) { cpSync(workflowSmoke,join(temp,'smoke'),{recursive:true});command(process.env.JS_BINARY__NODE_BINARY || process.execPath,['smoke/index.cjs'],temp); }
    if (json(join(temp,'node_modules',manifest.name,'package.json')).version !== JSON.parse((await archiveFiles(archive)).get('package/package.json')!.toString()).version) throw new Error('installed version differs');
  } finally { rmSync(temp, { recursive: true, force: true }); }
}
export async function main(): Promise<void> {
  const { positionals: [action], options: o } = argumentsOf();
  const project = resolve(root(), o['plugin-dir'] || o.project);
  if (action === 'pack') await pack(project, resolve(root(), o.compiled), resolve(root(), o.output), o.version || '0.0.0', o.declarations ? resolve(root(), o.declarations) : undefined);
  else if (action === 'check') await checkPackage(project, resolve(root(), o.archive), o.version);
  else if (action === 'smoke') await smoke(project, resolve(root(), o.archive));
  else throw new Error('expected pack, check, or smoke');
}
if (require.main === module) main().catch(e => { console.error(e.message); process.exitCode = 1; });
