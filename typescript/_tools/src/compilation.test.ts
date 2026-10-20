import assert from 'node:assert/strict';
import { cpSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join, resolve } from 'node:path';
import test from 'node:test';
import ts from 'typescript';

test('module compilation excludes neighboring sources and uses dependency declarations', () => {
  const directory = mkdtempSync(join(tmpdir(), 'typescript-compilation-'));
  try {
    // Windows actions share a directory containing other targets' sources.
    cpSync(resolve(__dirname, '../testdata/compilation'), directory, { recursive: true, dereference: true });
    writeFileSync(join(directory, 'src/unrelated.ts'), 'const invalid: number = "unrelated";');
    const configPath = join(directory, 'tsconfig_consumer_compile.json');
    const config = ts.readConfigFile(configPath, ts.sys.readFile);
    assert.equal(config.error, undefined);
    const parsed = ts.parseJsonConfigFileContent(config.config, ts.sys, directory, undefined, configPath);
    assert.deepEqual(parsed.fileNames.map(file => resolve(file)), [join(directory, 'src/consumer.ts')]);
    const program = ts.createProgram({
      rootNames: parsed.fileNames,
      options: parsed.options,
      projectReferences: parsed.projectReferences,
    });
    const diagnostics = [...parsed.errors, ...ts.getPreEmitDiagnostics(program)];
    assert.deepEqual(diagnostics.map(diagnostic => ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n')), []);
    const emitted: string[] = [];
    const result = program.emit(undefined, file => emitted.push(basename(file)));
    assert.equal(result.emitSkipped, false);
    assert.deepEqual(emitted.filter(file => !file.endsWith('.tsbuildinfo')).sort(), ['consumer.d.ts', 'consumer.js', 'consumer.js.map']);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
