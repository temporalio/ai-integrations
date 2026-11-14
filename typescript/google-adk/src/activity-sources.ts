/**
 * Lets the plugin refuse an Activity name that is already taken. The SDK's
 * plugin merge (`{...existing, ...param}`) keeps the last Activity of each
 * name, so without this check a plugin Activity silently replaces a different
 * Activity with the same name, or a later plugin silently replaces it. @internal
 */
import type { Worker } from '@temporalio/worker';

/**
 * What each plugin Activity is: the name it was built under (its operation)
 * followed by what it depends on. Two Activities with the same operation and
 * the same dependencies do the same thing, so one may replace the other.
 */
const activitySources = new WeakMap<object, readonly unknown[]>();

/**
 * Records, for every Activity in `activities`, its own key and `sources`, and
 * returns `activities`. The key keeps one operation, registered under another
 * operation's name, from passing as that operation.
 */
export function withSources<T extends object>(activities: T, sources: readonly unknown[]): T {
  for (const [operation, fn] of Object.entries(activities)) activitySources.set(fn, [operation, ...sources]);
  return activities;
}

/**
 * Throws if an Activity in `added` would replace a different Activity with the
 * same name in `existing`. The same function, or one built for the same
 * operation from the same sources with the same Activity metadata, may replace
 * itself: that is the same plugin instance, or one with the same options,
 * registered twice (e.g. on the NativeConnection and the Worker).
 */
export function refuseReplacedActivities(plugin: string, existing: object | undefined, added: object): void {
  // What the Worker registers: own enumerable keys with function values.
  const registered = new Map(Object.entries(existing ?? {}).filter(([, fn]) => typeof fn === 'function'));
  for (const [name, fn] of Object.entries(added)) {
    const other = registered.get(name);
    if (other === undefined || other === fn || sameSources(other, fn)) continue;
    throw new TypeError(
      `Duplicate registration of Activity '${name}': ${plugin} would replace a different Activity with this name. ` +
        remedy(plugin)
    );
  }
}

/**
 * For `runWorker`, which runs after every plugin's `configureWorker`: if a
 * plugin applied after this one replaced an Activity in `added` with a
 * different one, returns a promise that rejects with a `TypeError`. Before
 * rejecting, it stops the Worker so that it never polls, and lets `next`
 * release it as `run()` does, so `connection.close()` still works.
 *
 * Otherwise it returns `undefined` without yielding, so the caller can start
 * the Worker in the same tick. `Worker.runUntil` calls its callback right
 * after `run()` and stops the Worker afterwards only if it is already running;
 * a Worker started later polls until something else stops it.
 */
export function refuseLaterReplacement(
  plugin: string,
  worker: Worker,
  added: object,
  next: (worker: Worker) => Promise<void>
): Promise<never> | undefined {
  for (const [name, fn] of Object.entries(added)) {
    const kept = worker.options.activities.get(name);
    if (kept === undefined || kept === fn || sameSources(kept, fn)) continue;
    const error = new TypeError(
      `Duplicate registration of Activity '${name}': a plugin applied after ${plugin} replaced its Activity with this name. ` +
        remedy(plugin)
    );
    // There is no public way to stop a Worker that has not run yet.
    (worker as unknown as { nativeWorker: { initiateShutdown(): void } }).nativeWorker.initiateShutdown();
    const fail = (): never => {
      throw error;
    };
    return next(worker).then(fail, fail);
  }
  return undefined;
}

function remedy(plugin: string): string {
  return (
    'Rename or remove the other Activity, run the other integration on its own Worker and task queue, ' +
    `or use one ${plugin} instance on this Worker.`
  );
}

function sameSources(a: object, b: object): boolean {
  const x = activitySources.get(a);
  const y = activitySources.get(b);
  return (
    x !== undefined &&
    y !== undefined &&
    x.length === y.length &&
    x.every((v, i) => v === y[i]) &&
    activityOptions(a) === activityOptions(b)
  );
}

/**
 * The metadata `setActivityOptions` attached to `fn`, read the way the Worker
 * reads it (an own object-valued `activityDefinitionOptions`). Its `typeInfo`
 * decodes the Activity's arguments and encodes its result, so a function with
 * other metadata is a different Activity even when its sources match.
 */
function activityOptions(fn: object): object | undefined {
  if (!Object.hasOwn(fn, 'activityDefinitionOptions')) return undefined;
  const options: unknown = (fn as { activityDefinitionOptions?: unknown }).activityDefinitionOptions;
  return typeof options === 'object' && options !== null ? options : undefined;
}
