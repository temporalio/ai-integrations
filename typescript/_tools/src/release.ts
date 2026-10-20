/** npm release policy, exact-byte verification, and recovery of tested artifacts. */
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { createHash } from 'node:crypto';
import * as semver from 'semver';
import { archiveFiles } from './archive';
import { argumentsOf, command, Document, fetchDocument, gh, git, json, outputs, REPOSITORY, root, toml } from './common';

const FIRST: Record<string, string> = { 'generally-available': '1.0.0', 'public-preview': '0.1.0', 'pre-release': '0.0.1' };
export function version(raw: string): semver.SemVer {
  const parsed = semver.parse(raw);
  if (!parsed || parsed.version !== raw || parsed.build.length) throw new Error('npm version must be canonical SemVer without build metadata');
  return parsed;
}
export function parseTag(tag: string): Document {
  const match = /^typescript\/([a-z0-9][a-z0-9_.-]*)\/v(.+)$/.exec(tag);
  if (!match || match[1]!.includes('..')) throw new Error('expected typescript/<plugin>/v<version>');
  const v = version(match[2]!);
  return { tag, language: 'typescript', plugin: match[1], version: match[2], prerelease: String(v.prerelease.length > 0), plugin_dir: `typescript/${match[1]}` };
}
export function checkPolicy(raw: string, maturity: string, published: string[]): boolean {
  const v = version(raw), first = FIRST[maturity];
  if (!first) throw new Error(`unknown maturity ${maturity}`);
  if (!published.length) {
    if ([v.major, v.minor, v.patch].join('.') !== first) throw new Error(`first release must be ${first}`);
    return false;
  }
  const highest = published.map((item) => version(item)).sort(semver.rcompare)[0]!;
  if (semver.lt(v, highest)) throw new Error(`version must advance past ${highest.version}`);
  return semver.eq(v, highest);
}
export async function fetchPackage(coordinate: string, local?: string): Promise<Document | undefined> {
  const result = await fetchDocument(`https://registry.npmjs.org/${encodeURIComponent(coordinate)}`, local);
  if (result && (result.name !== coordinate || !result.versions || typeof result.versions !== 'object')) throw new Error('npm returned invalid package metadata');
  return result;
}
export async function artifact(dist: string, coordinate: string, raw: string): Promise<string> {
  dist = resolve(root(), dist);
  const files = readdirSync(dist).filter((name) => name.endsWith('.tgz'));
  if (files.length !== 1) throw new Error('expected exactly one tested npm tarball');
  const path = join(dist, files[0]!);
  const manifest = (await archiveFiles(path)).get('package/package.json');
  if (!manifest) throw new Error('tarball must contain package/package.json');
  const pkg = JSON.parse(manifest.toString());
  if (pkg.name !== coordinate || pkg.version !== raw || pkg.private) throw new Error('tarball identity or private flag differs from release');
  return path;
}
export function verifyArtifact(metadata: Document, payload: Buffer, coordinate: string, raw: string): void {
  const integrity = 'sha512-' + createHash('sha512').update(payload).digest('base64');
  if (metadata.name !== coordinate || metadata.version !== raw || metadata.deprecated || metadata.dist?.integrity !== integrity) {
    throw new Error('npm identity, deprecation state, or integrity differs from the tested tarball');
  }
}
export async function registryState(coordinate: string, raw: string, dist: string): Promise<boolean> {
  const tarball = await artifact(dist, coordinate, raw);
  const pkg = await fetchPackage(coordinate);
  if (!pkg?.versions[raw]) return false;
  verifyArtifact(pkg.versions[raw], readFileSync(tarball), coordinate, raw);
  return true;
}
export function checkRecoveryRun(tag: string, runId: number, repo = REPOSITORY): void {
  const parsed = parseTag(tag);
  if (!Number.isSafeInteger(runId) || runId <= 0) throw new Error('recovery requires a positive run ID');
  const sha = git('rev-parse', '--verify', `refs/tags/${tag}^{commit}`);
  git('merge-base', '--is-ancestor', sha, 'origin/main');
  for (const name of ['plugin.toml', 'package.json', 'pnpm-lock.yaml', 'pnpm-workspace.yaml']) {
    const relative = `${parsed.plugin_dir}/${name}`;
    if (command('git', ['show', `refs/tags/${tag}:${relative}`]) !== readFileSync(resolve(root(), relative), 'utf8')) throw new Error(`recovery metadata differs from tag: ${relative}`);
  }
  const run = gh(`repos/${repo}/actions/runs/${runId}`);
  const sourceMatches = run.event === 'release' ? run.head_branch === tag : run.event === 'workflow_dispatch' && ['main', tag].includes(run.head_branch);
  if (run.repository?.full_name !== repo || run.path !== '.github/workflows/release-typescript.yml' || !sourceMatches || run.head_sha !== sha || run.status !== 'completed') {
    throw new Error('recovery source must be a completed release or main/tag dry run at the exact tagged commit');
  }
  const metadata = toml(resolve(root(), parsed.plugin_dir, 'plugin.toml'));
  const versions = metadata.ci['runtime-versions'] as string[];
  const cells = new Set([`ubuntu-latest, node${versions[0]}`, `ubuntu-latest, node${versions.at(-1)}`, `macos-latest, node${versions.at(-1)}`, `windows-latest, node${versions.at(-1)}`]);
  const required = ['Validate TypeScript tag and version policy', 'Test / matrix', ...[...cells].map((cell) => `Test / ${parsed.plugin} (${cell})`)];
  const pages = gh(`repos/${repo}/actions/runs/${runId}/jobs?per_page=100`, '--paginate', '--slurp');
  const jobs = new Map(pages.flatMap((page: Document) => page.jobs).map((job: Document) => [job.name, job]));
  for (const name of required) {
    const job = jobs.get(name) as Document | undefined;
    if (job?.status !== 'completed' || job.conclusion !== 'success') throw new Error(`recovery source has no successful job: ${name}`);
  }
  const name = `npm-dist-${parsed.plugin}`;
  const items = gh(`repos/${repo}/actions/runs/${runId}/artifacts?name=${name}&per_page=100`).artifacts
    .filter((item: Document) => item.name === name && !item.expired);
  if (items.length !== 1) throw new Error('recovery requires exactly one unexpired distribution artifact');
}
export function releaseNotes(pluginDir: string, tag: string, ref = tag, repo = REPOSITORY): string {
  const parsed = parseTag(tag), current = version(parsed.version);
  const previous = git('tag', '--list', `typescript/${parsed.plugin}/v*`).split('\n').filter(Boolean)
    .map((value) => ({ tag: value, version: version(value.split('/v').at(-1)!) }))
    .filter((item) => semver.lt(item.version, current)).sort((a, b) => semver.rcompare(a.version, b.version))[0];
  const metadata = toml(resolve(root(), pluginDir, 'plugin.toml'));
  const subjects = git('log', '--no-decorate', '--format=%h%x1f%s', previous ? `${previous.tag}..${ref}` : ref, '--', pluginDir);
  const lines = previous ? [] : [`First standalone release of \`${metadata.plugin.coordinate}\`.`, ''];
  lines.push("## What's changed", '');
  for (const line of subjects.split('\n').filter(Boolean)) {
    const [sha, subject] = line.split('\x1f');
    lines.push(`- ${subject!.replace(/(?<![\w/#])#(\d+)\b/g, `[#$1](https://github.com/${repo}/pull/$1)`)} (${sha})`);
  }
  if (!subjects) lines.push('_No commits touched this plugin since the previous release._');
  lines.push('', previous ? `**Full changelog**: https://github.com/${repo}/compare/${previous.tag}...${tag}` : `**Source**: https://github.com/${repo}/tree/refs/tags/${tag}/${pluginDir}`, '');
  if (current.prerelease.length) lines.push('## Pre-release notes', '', `Install this npm pre-release with \`npm install ${metadata.plugin.coordinate}@${current.version}\`.`, '');
  return lines.join('\n');
}
export async function main(argv?: string[]): Promise<void> {
  const { positionals: [action, tag], options: o } = argumentsOf(argv);
  if (o['repo-root']) process.env.BUILD_WORKSPACE_DIRECTORY = resolve(o['repo-root']);
  const repo = o.repo || REPOSITORY;
  if (action === 'parse-tag') {
    const parsed = parseTag(tag!);
    const metadata = toml(resolve(root(), parsed.plugin_dir, 'plugin.toml'));
    const pkg = json(resolve(root(), parsed.plugin_dir, 'package.json'));
    const coordinate = `@temporalio/${parsed.plugin}`;
    if (metadata.plugin.name !== parsed.plugin || metadata.plugin.language !== 'typescript' || metadata.plugin.registry !== 'npm' || metadata.plugin.coordinate !== coordinate || pkg.name !== coordinate) throw new Error('tag, metadata and package must agree');
    if (!FIRST[metadata.plugin.maturity]) throw new Error('unknown maturity');
    outputs({ ...parsed, coordinate, root_api: metadata.plugin['root-api'], smoke_imports: metadata.smoke.imports.join(','), maturity: metadata.plugin.maturity, github_prerelease: String(parsed.prerelease === 'true' || metadata.plugin.maturity === 'pre-release'), dist_tag: parsed.prerelease === 'true' ? 'next' : 'latest' }, o['github-output']);
  } else if (action === 'check-version-policy') {
    const metadata = toml(resolve(root(), o['plugin-dir'], 'plugin.toml'));
    const pkg = await fetchPackage(metadata.plugin.coordinate, o['registry-json']);
    if (checkPolicy(o.version, metadata.plugin.maturity, Object.keys(pkg?.versions || {}))) console.warn('::warning::newest version already published; verifying a recovery');
  } else if (action === 'npm-registry-state') {
    version(o.version);
    const attempts = Number(o.attempts || 1), delay = Number(o.delay || 10);
    if (!Number.isInteger(attempts) || attempts < 1 || delay < 0) throw new Error('invalid verification retry settings');
    let failure: unknown;
    for (let attempt = 0; attempt < attempts; attempt++) {
      try {
        const exists = await registryState(o.coordinate, o.version, o.dist);
        if (exists || !o['require-present']) { outputs({ exists: String(exists) }, o['github-output']); return; }
        failure = new Error('npm release has not propagated');
      } catch (error) { failure = error; }
      if (attempt + 1 < attempts) await new Promise((done) => setTimeout(done, delay * 1000));
    }
    throw failure;
  } else if (action === 'check-recovery-run') {
    checkRecoveryRun(o.tag, Number(o['run-id']), repo);
    outputs({ recovery: 'true', artifact_run_id: o['run-id'] }, o['github-output']);
  } else if (action === 'check-published-release') {
    const parsed = parseTag(o.tag);
    const maturity = toml(resolve(root(), parsed.plugin_dir, 'plugin.toml')).plugin.maturity;
    if (!FIRST[maturity]) throw new Error('unknown maturity');
    const release = gh(`repos/${repo}/releases/tags/${encodeURIComponent(o.tag)}`);
    if (release.tag_name !== o.tag || release.draft !== false || release.prerelease !== (parsed.prerelease === 'true' || maturity === 'pre-release') || !release.html_url) throw new Error('published GitHub release identity or classification differs');
    outputs({ url: release.html_url }, o['github-output']);
  } else if (action === 'release-notes') {
    writeFileSync(resolve(root(), o.output), releaseNotes(o['plugin-dir'], o.tag, o.ref || o.tag, repo));
  } else throw new Error(`unknown TypeScript release operation ${action}`);
}

if (require.main === module) main().catch((error) => { console.error(error.message); process.exitCode = 1; });
