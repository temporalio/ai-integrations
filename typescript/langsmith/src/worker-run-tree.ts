/** Scoped LangSmith defaults for worker handlers with no propagated parent. */

import { AsyncLocalStorage } from 'node:async_hooks';
import { RunTree, type RunTreeConfig } from 'langsmith/run_trees';
import { AsyncLocalStorageProviderSingleton } from 'langsmith/singletons/traceable';
import type { EmitterConfig } from './sinks';

const context = new AsyncLocalStorage<EmitterConfig>();
let installed = false;

interface AddressingHook {
  _resolveAddressing(config: RunTreeConfig): void;
}

function installDefaults(): void {
  if (installed) return;
  installed = true;
  const prototype = RunTree.prototype as unknown as AddressingHook;
  const originalAddressing = prototype._resolveAddressing;
  const originalClient = RunTree.getSharedClient;

  RunTree.getSharedClient = function () {
    return context.getStore()?.client ?? originalClient.call(this);
  };
  prototype._resolveAddressing = function (this: RunTree, config) {
    const defaults = context.getStore();
    if (!defaults) return originalAddressing.call(this, config);
    this.replicas ??= defaults.replicas;
    // Explicit destinations and native child inheritance outrank the plugin.
    const selected = config.project_name || config.address != null
      ? config
      : { ...config, project_name: defaults.projectName, address: defaults.address };
    return originalAddressing.call(this, selected);
  };
}

/** Supply native roots with defaults without installing an ambient RunTree. */
export function withRunTreeDefaults<T>(config: EmitterConfig, fn: () => T): T {
  installDefaults();
  const storage = AsyncLocalStorageProviderSingleton.getInstance() as {
    run<T>(store: undefined, fn: () => T): T;
  };
  return context.run(config, () => storage.run(undefined, fn));
}
