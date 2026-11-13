import path from 'node:path';
import test from 'ava';
import { InMemoryRunCollector, runAndReplay, useSharedEnv } from './helpers';
import { EsmTraceableWorkflow } from './workflows/esm-traceable';

process.env.LANGSMITH_TRACING = 'true';
// The TypeScript source, not the CommonJS build, so langsmith is imported through its ESM `import` condition.
const WORKFLOWS_PATH = path.resolve(__dirname, '../../src/__tests__/workflows/esm-traceable.ts');
useSharedEnv(test);

test('traceable in an ESM workflow uses the patched langsmith build', async (t) => {
  const collector = new InMemoryRunCollector();
  const { ids } = await runAndReplay(t, collector, WORKFLOWS_PATH, EsmTraceableWorkflow);
  t.deepEqual([collector.byName('esm-root')?.id, collector.byName('esm-child')?.id], ids);
});
