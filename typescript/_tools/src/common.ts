/** Native TypeScript tooling primitives. No Python or another plugin's environment. */
import { appendFileSync, readFileSync, readdirSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { parse } from 'smol-toml';

export type Document = Record<string, any>;
export const REPOSITORY = process.env.GITHUB_REPOSITORY || 'temporalio/ai-integrations';
export const root = () => resolve(process.env.BUILD_WORKSPACE_DIRECTORY || process.env.GITHUB_WORKSPACE || '.');
export const json = (file: string): Document => JSON.parse(readFileSync(file, 'utf8'));
export const toml = (file: string): Document => parse(readFileSync(file, 'utf8'));
export const git = (...args: string[]) => execFileSync('git', args, { cwd: root(), encoding: 'utf8' }).trimEnd();
export function gh(...args: string[]): any {
  return JSON.parse(execFileSync('gh', ['api', ...args], { encoding: 'utf8' }));
}
export function command(file: string, args: string[], cwd = root()): string {
  return execFileSync(file, args, { cwd, encoding: 'utf8', stdio: ['pipe', 'pipe', 'inherit'] });
}
export function outputs(values: Document, file = process.env.GITHUB_OUTPUT): void {
  for (const [key, input] of Object.entries(values)) {
    const value = typeof input === 'string' ? input : JSON.stringify(input);
    if (/[\r\n]/.test(value)) throw new Error('GitHub output contains a line break');
    const line = `${key}=${value}\n`;
    process.stdout.write(line);
    if (file) appendFileSync(file, line);
  }
}
export function argumentsOf(argv = process.argv.slice(2)): { positionals: string[]; options: Document } {
  const positionals: string[] = [], options: Document = {};
  const booleans = new Set(['require-present', 'dry-run', 'prerelease', 'latest']);
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;
    if (!arg.startsWith('--')) positionals.push(arg);
    else {
      const [key, ...value] = arg.slice(2).split('=');
      if (value.length) options[key!] = value.join('=');
      else if (booleans.has(key!)) options[key!] = true;
      else {
        if (!argv[i + 1] || argv[i + 1]!.startsWith('--')) throw new Error(`missing value for ${arg}`);
        options[key!] = argv[++i];
      }
    }
  }
  return { positionals, options };
}
export function plugins(): Document[] {
  return readdirSync(join(root(), 'typescript'), { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && !entry.name.startsWith('_') && !entry.name.startsWith('.'))
    .map((entry) => ({ path: `typescript/${entry.name}`, ...toml(join(root(), 'typescript', entry.name, 'plugin.toml')) }));
}
export async function fetchDocument(url: string, local?: string): Promise<Document | undefined> {
  if (local) {
    try { return json(resolve(root(), local)); } catch (error: any) { if (error.code === 'ENOENT') return undefined; throw error; }
  }
  const response = await fetch(url, { signal: AbortSignal.timeout(30_000) });
  if (response.status === 404) return undefined;
  if (!response.ok) throw new Error(`HTTP ${response.status} from registry; refusing to guess`);
  return response.json() as Promise<Document>;
}
