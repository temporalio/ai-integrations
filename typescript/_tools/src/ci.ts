import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { argumentsOf, git, json, outputs, plugins, root } from './common';
export function selected(files: string[], names: string[]): string[] {
  const result = new Set<string>();
  for (const file of files) {
    if (/^(\.github\/|bazel\/|MODULE\.bazel|BUILD\.bazel|REPO\.bazel|\.bazel|LICENSE$)/.test(file)) return names;
    if (!file.startsWith('typescript/')) continue;
    const name = file.split('/')[1]!;
    if (names.includes(name)) result.add(name); else names.forEach(n => result.add(n));
  }
  return [...result].sort();
}
export async function main(): Promise<void> {
  const { positionals: [action], options: o } = argumentsOf(), all = plugins();
  if (action === 'matrix') {
    const meta = all.find(p => p.path === o['plugin-dir']); if (!meta) throw new Error('unknown plugin');
    const versions: string[] = meta.ci['runtime-versions'], min=versions[0]!, max=versions[versions.length-1]!;
    outputs({ matrix: { include: [...new Set([min,max])].map(runtime => ({os:'ubuntu-latest',runtime,dist:runtime===max})).concat(['macos-latest','windows-latest'].map(os => ({os,runtime:max,dist:false}))) } }, o['github-output']);
  } else if (action === 'select') {
    const names = all.map(p => p.plugin.name).sort(); let result = names;
    if (process.env.GITHUB_EVENT_NAME === 'pull_request' || o.base) { try { result=selected(git('diff','--name-only','--no-renames',`${o.base || `origin/${process.env.GITHUB_BASE_REF || 'main'}`}...HEAD`).split('\n'),names); } catch { /* fail open to full matrix */ } }
    outputs({typescript:result},o['github-output']);
  } else if (action === 'conventions') {
    for (const meta of all) {
      const dir=join(root(),meta.path), m=json(join(dir,'package.json')), name=meta.plugin.name;
      if (name!==meta.path.split('/')[1] || meta.plugin.language!=='typescript' || meta.plugin.coordinate!==`@temporalio/${name}` || m.name!==meta.plugin.coordinate || m.version!=='0.0.0' || m.license!=='MIT' || m.packageManager!=='pnpm@10.34.5') throw new Error(`invalid TypeScript metadata: ${meta.path}`);
      for (const file of ['BUILD.bazel','LICENSE','README.md','pnpm-lock.yaml','pnpm-workspace.yaml','tsconfig.json']) if (!existsSync(join(dir,file))) throw new Error(`missing ${dir}/${file}`);
      if (!readFileSync(join(dir,'LICENSE')).equals(readFileSync(join(root(),'LICENSE')))) throw new Error('LICENSE differs');
      for (const value of Object.values({...m.dependencies,...m.peerDependencies})) if (/^(link:|file:|workspace:)/.test(String(value))) throw new Error('cross-plugin local dependency');
    }
  } else throw new Error(`unknown TypeScript CI operation: ${action}`);
}
if (require.main===module) main().catch(e => {console.error(e.message);process.exitCode=1;});
