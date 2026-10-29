"""Prepare the pinned checkout; change only disposable project files and dependencies."""
import json
from pathlib import Path
import shutil
import sys

plugin = Path(sys.argv[1]).resolve()
mode = sys.argv[2]
evidence = Path(__file__).resolve().parent
saved = plugin / '.boundary-originals'
saved.mkdir(exist_ok=True)
paths = ['src/workflow/runner.ts', 'src/workflow/trace-interceptor.ts']
for relative in paths:
    source = plugin / relative
    backup = saved / source.name
    if not backup.exists():
        shutil.copyfile(source, backup)
    shutil.copyfile(backup, source)

core = (plugin / 'node_modules/@openai/agents-core').resolve()
duplicate = core.with_name('agents-core-copy')
if not duplicate.exists():
    shutil.copytree(core, duplicate, symlinks=True)
link = plugin / 'node_modules/@openai/agents-core-copy'
if not link.exists():
    link.symlink_to(duplicate, target_is_directory=True)
zod = (core.parent.parent / 'zod').resolve()
zod_link = plugin / 'node_modules/boundary-zod'
if not zod_link.exists():
    zod_link.symlink_to(zod, target_is_directory=True)

for package in (core, duplicate):
    label = 'core' if package == core else 'duplicate'
    manifest = package / 'package.json'
    manifest_backup = saved / (label + '-package.json')
    if not manifest_backup.exists():
        shutil.copyfile(manifest, manifest_backup)
    metadata = json.loads(manifest_backup.read_text())
    if mode == 'policy':
        metadata['exports']['./runtime-policy'] = {
            'types': './dist/runtimePolicy.d.ts', 'require': './dist/runtimePolicy.js'}
        metadata.setdefault('typesVersions', {}).setdefault('*', {})['runtime-policy'] = ['dist/runtimePolicy.d.ts']
        (package / 'dist/runtimePolicy.js').write_text("""const key = Symbol.for('@openai/agents/runtimePolicy/v1');
exports.setRuntimePolicy = function setRuntimePolicy(policy) {
    Object.defineProperty(globalThis, key, { value: policy, configurable: true });
};
""")
        (package / 'dist/runtimePolicy.d.ts').write_text(
            'export declare function setRuntimePolicy(policy: { prepareRun(runner: any, agent: any, options: any): any; '
            'resolveModel(runner: any, agent: any, selected: any): any }): void;\n')
    manifest.write_text(json.dumps(metadata, indent=2) + '\n')
    run = package / 'dist/run.js'
    backup = saved / ('core-run.js' if package == core else 'duplicate-run.js')
    if not backup.exists():
        shutil.copyfile(run, backup)
    original = backup.read_text()
    updated = original
    if mode == 'policy':
        marker = '    }) {\n        if (input instanceof runState_1.RunState) {'
        assert original.count(marker) == 1
        updated = updated.replace(marker, '''    }) {
        const runtimePolicy = globalThis[Symbol.for('@openai/agents/runtimePolicy/v1')];
        if (runtimePolicy) options = runtimePolicy.prepareRun(this, agent, options);
        if (input instanceof runState_1.RunState) {''')
        marker = """        const resolvedModel = typeof selectedModel === 'string'
            ? await this.config.modelProvider.getModel(selectedModel)
            : selectedModel;"""
        assert original.count(marker) == 1
        updated = updated.replace(marker, """        const runtimePolicy = globalThis[Symbol.for('@openai/agents/runtimePolicy/v1')];
        const resolvedModel = runtimePolicy
            ? await runtimePolicy.resolveModel(this, agent, selectedModel)
            : typeof selectedModel === 'string'
                ? await this.config.modelProvider.getModel(selectedModel)
                : selectedModel;""")
    run.write_text(updated)

if mode == 'policy':
    runner = plugin / paths[0]
    runner.write_text(runner.read_text().replace("import { checkAgent } from './check-agent';\n", '')
                      .replace('    checkAgent(agent, modelOverride);\n', ''))
    tracing = plugin / paths[1]
    tracing.write_text(tracing.read_text().replace("import { installAgentToolInterception } from './agent-tools';",
                                                  "import './runtime-policy';")
                       .replace('installAgentToolInterception();', ''))
    shutil.copyfile(evidence / 'runtime-policy.ts', plugin / 'src/workflow/runtime-policy.ts')
else:
    (plugin / 'src/workflow/runtime-policy.ts').unlink(missing_ok=True)

for name in ('proof.cjs', 'workflows.cjs', 'replay.cjs', 'require-runtime-hooks.cjs', 'upstream.cjs'):
    shutil.copyfile(evidence / name, plugin / ('boundary-' + name))
print(json.dumps({'mode': mode, 'core': str(core), 'duplicate': str(duplicate),
                  'version': json.loads((core / 'package.json').read_text())['version']}))
