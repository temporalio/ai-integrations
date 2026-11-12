import { address } from 'langsmith';
import { RunTree } from 'langsmith/run_trees';
import { getCurrentRunTree, traceable, withRunTree } from 'langsmith/traceable';
import { proxySinks, sleep } from '@temporalio/workflow';

function snapshot(run: RunTree): Record<string, unknown> {
  return {
    id: run.id,
    trace_id: run.trace_id,
    parent_run_id: run.parent_run?.id ?? run.parent_run_id ?? null,
    dotted_order: run.dotted_order,
    start_time: run.start_time,
    end_time: run.end_time,
    native: Object.getPrototypeOf(run) === RunTree.prototype,
  };
}

export async function NativeRunTreeWorkflow(): Promise<Record<string, unknown>[]> {
  const root = new RunTree({
    name: 'native-root',
    project_name: 'native-project',
    replicas: [{ projectName: 'primary', primary: true }, { projectName: 'secondary' }],
    metadata: { inherited: true },
  });
  await root.postRun();
  const child = root.createChild({ name: 'native-child', project_name: 'ignored-child-project' });
  await child.postRun();
  await sleep(1);
  await child.end({ ok: true }, undefined, undefined, { completed: true });
  await child.patchRun();
  let tracedChild: RunTree | undefined;
  await withRunTree(root, () => traceable(async () => {
    tracedChild = getCurrentRunTree();
    await sleep(1);
  }, { name: 'traceable-native-child' })());
  await root.end({ ok: true });
  await root.patchRun();
  let tracedRoot: RunTree | undefined;
  await traceable(async () => {
    tracedRoot = getCurrentRunTree();
    await sleep(1);
  }, { name: 'traceable-root', processInputs: async () => ({ processed: true }) })();
  const explicit = new RunTree({
    name: 'explicit-root',
    id: '12345678-1234-4234-8234-123456789012',
    start_time: 1234567890000,
  });
  await explicit.postRun();
  await explicit.end({}, undefined, 1234567891000);
  await explicit.patchRun({ excludeInputs: false });
  const snapshots = [root, child, tracedChild!, tracedRoot!, explicit].map(snapshot);
  try {
    new RunTree({ name: 'conflicting-destinations', project_name: 'project', address: address.agent('agent', 'env') });
    snapshots[0]!.conflictRejected = false;
  } catch {
    snapshots[0]!.conflictRejected = true;
  }
  proxySinks<{ nativeRuns: { capture(runs: Record<string, unknown>[]): void } }>().nativeRuns.capture(snapshots);
  return snapshots;
}
