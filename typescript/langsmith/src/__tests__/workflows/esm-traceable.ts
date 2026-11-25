// Bundled from this TypeScript source (see test-esm-workflow.ts) so webpack sees real ESM `import`s.
import { getCurrentRunTree, traceable } from 'langsmith/traceable';
import { proxySinks, sleep } from '@temporalio/workflow';

export async function EsmTraceableWorkflow(): Promise<string[]> {
  const ids: string[] = [];
  await traceable(async () => {
    ids.push(getCurrentRunTree().id);
    await sleep(1);
    await traceable(async () => {
      ids.push(getCurrentRunTree().id);
    }, { name: 'esm-child' })();
  }, { name: 'esm-root' })();
  proxySinks<{ runIds: { capture(ids: string[]): void } }>().runIds.capture(ids);
  return ids;
}
