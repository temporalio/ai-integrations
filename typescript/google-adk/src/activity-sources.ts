/**
 * Lets the plugin refuse an Activity name that is already taken. The SDK's
 * plugin merge (`{...existing, ...param}`) keeps the last Activity of each
 * name, so without this check a plugin Activity silently replaces a different
 * Activity with the same name. @internal
 */

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
        `Rename or remove the other registration, or give each ${plugin} on this Worker the same options.`
    );
  }
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
