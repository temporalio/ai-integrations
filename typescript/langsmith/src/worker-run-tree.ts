/** Scoped LangSmith defaults for worker handlers with no propagated parent. */

import { AsyncLocalStorage } from 'node:async_hooks';
import type { RunTree, RunTreeConfig } from 'langsmith/run_trees';
import { AsyncLocalStorageProviderSingleton } from 'langsmith/singletons/traceable';
import { patchRunTree, type RunTreeOverrides } from './run-tree';
import type { EmitterConfig } from './sinks';

const context = new AsyncLocalStorage<EmitterConfig>();
let installed = false;

export const WORKER_RUN_TREE_OVERRIDES: RunTreeOverrides = {
  statics: {
    getSharedClient: (original) =>
      function (this: typeof RunTree, ...args: unknown[]) {
        return context.getStore()?.client ?? original.apply(this, args);
      },
  },
  methods: {
    _resolveAddressing: (original) =>
      function (this: RunTree, config: RunTreeConfig, ...args: unknown[]) {
        const defaults = context.getStore();
        if (!defaults) return original.call(this, config, ...args);
        this.replicas ??= defaults.replicas;
        // Explicit destinations and native child inheritance outrank the plugin.
        const selected = config.project_name || config.address != null
          ? config
          : { ...config, project_name: defaults.projectName, address: defaults.address };
        return original.call(this, selected, ...args);
      },
  },
};

function installDefaults(): void {
  if (installed) return;
  patchRunTree(WORKER_RUN_TREE_OVERRIDES);
  installed = true;
}

/** Supply native roots with defaults without installing an ambient RunTree. */
export function withRunTreeDefaults<T>(config: EmitterConfig, fn: () => T): T {
  installDefaults();
  const storage = AsyncLocalStorageProviderSingleton.getInstance() as {
    run<T>(store: undefined, fn: () => T): T;
  };
  return context.run(config, () => storage.run(undefined, fn));
}
