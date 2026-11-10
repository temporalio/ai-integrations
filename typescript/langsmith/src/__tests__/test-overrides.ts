/**
 * The plugin's LangSmith overrides forward every argument, and Worker setup and the Workflow install refuse to run,
 * patching nothing, when a LangSmith member they wrap is missing.
 *
 * @module
 */

import test from 'ava';
import { __version__ } from 'langsmith';
import { RunTree } from 'langsmith/run_trees';
import type { WorkerOptions } from '@temporalio/worker';

import { LangSmithPlugin } from '../plugin';
import type { RunTreeOverrides } from '../run-tree';
import { WORKER_RUN_TREE_OVERRIDES, withRunTreeDefaults } from '../worker-run-tree';
import { WORKFLOW_RUN_TREE_OVERRIDES, installRunTreeOverrides, workflowRunTreeContext } from '../workflow-run-tree';
import { InMemoryRunCollector } from './helpers';

type Members = Record<string, unknown>;
const statics = RunTree as unknown as Members;
const methods = RunTree.prototype as unknown as Members;

function snapshot(): Map<Members, Members> {
  return new Map(
    [statics, methods].map((owner) => [
      owner,
      Object.fromEntries(
        Object.getOwnPropertyNames(owner).map((name) => [name, Object.getOwnPropertyDescriptor(owner, name)!.value])
      ),
    ])
  );
}

test.serial('a missing LangSmith member fails Worker setup and the Workflow install, patching nothing', (t) => {
  const plugin = new LangSmithPlugin({ client: new InMemoryRunCollector().asClient(), projectName: 'p' });
  const workerSetup = () => plugin.configureWorker({ taskQueue: 'tq' } as WorkerOptions);
  const cases = [
    { owner: statics, label: 'RunTree.getSharedClient', name: 'getSharedClient' },
    { owner: methods, label: 'RunTree.prototype.patchRun', name: 'patchRun' },
  ];
  for (const { owner, label, name } of cases) {
    const original = owner[name];
    delete owner[name];
    const before = snapshot();
    try {
      for (const install of [workerSetup, installRunTreeOverrides]) {
        t.throws(install, {
          message: `@temporalio/langsmith does not support LangSmith ${__version__}: missing ${label}`,
        });
      }
      for (const [target, members] of snapshot()) {
        for (const [member, value] of Object.entries(members)) t.is(value, before.get(target)![member], member);
      }
    } finally {
      owner[name] = original;
    }
  }
});

const EXTRA = Symbol('extra');

/** Whether `override`, wrapping a stub original, passes a trailing extra argument through to it. */
async function forwards(
  overrides: RunTreeOverrides,
  kind: keyof RunTreeOverrides,
  name: string,
  self: unknown,
  ...args: unknown[]
): Promise<boolean> {
  let received: unknown[] = [];
  const replacement = overrides[kind][name]!(function (...forwarded: unknown[]) {
    received = forwarded;
    return {};
  });
  await replacement.call(self, ...args, EXTRA);
  return received.at(-1) === EXTRA;
}

test.serial('overrides forward extra arguments to LangSmith inside and outside plugin contexts', async (t) => {
  const client = new InMemoryRunCollector().asClient();
  const run = new RunTree({ name: 'run', project_name: 'project', client });
  const outside: Array<[RunTreeOverrides, keyof RunTreeOverrides, string, unknown, ...unknown[]]> = [
    [WORKFLOW_RUN_TREE_OVERRIDES, 'statics', 'getDefaultConfig', RunTree],
    [WORKFLOW_RUN_TREE_OVERRIDES, 'statics', 'getSharedClient', RunTree],
    [WORKFLOW_RUN_TREE_OVERRIDES, 'methods', '_resolveAddressing', run, { project_name: 'project' }],
    [WORKFLOW_RUN_TREE_OVERRIDES, 'methods', 'end', run, {}, undefined, 1, {}],
    [WORKFLOW_RUN_TREE_OVERRIDES, 'methods', 'postRun', run, true],
    [WORKFLOW_RUN_TREE_OVERRIDES, 'methods', 'patchRun', run, { excludeInputs: false }],
    [WORKER_RUN_TREE_OVERRIDES, 'statics', 'getSharedClient', RunTree],
    [WORKER_RUN_TREE_OVERRIDES, 'methods', '_resolveAddressing', run, { project_name: 'project' }],
  ];
  for (const [overrides, kind, name, self, ...args] of outside) {
    t.true(await forwards(overrides, kind, name, self, ...args), `outside: ${name}`);
  }

  const workflowConfig = { addTemporalRuns: false, destination: { project_name: 'workflow-project' } };
  const self = new RunTree({ name: 'workflow', project_name: 'project', client });
  t.true(
    await workflowRunTreeContext(workflowConfig, () =>
      forwards(WORKFLOW_RUN_TREE_OVERRIDES, 'methods', '_resolveAddressing', self, {})
    ),
    'workflow: _resolveAddressing'
  );

  const workerConfig = { client, addTemporalRuns: false, projectName: 'p' };
  const forwarded = await withRunTreeDefaults(workerConfig, () =>
    forwards(WORKER_RUN_TREE_OVERRIDES, 'methods', '_resolveAddressing', {}, {})
  );
  t.true(forwarded, 'worker: _resolveAddressing');
});
