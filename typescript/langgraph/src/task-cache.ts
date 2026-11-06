/**
 * Task result cache for continue-as-new.
 *
 * Results are cached by `(activity name, arguments, context)` hash so that
 * nodes and tasks that already completed are not executed again after a
 * continue-as-new. The cache is a plain object that travels through the
 * continue-as-new input.
 *
 * This module is part of the Workflow-sandbox import graph: it must not import
 * any Worker-only module.
 *
 * @module
 * @internal
 */

import { sha256Hex } from './sha256';

/** The serializable task result cache returned by `cache()`. */
export type TaskCache = Record<string, unknown>;

function isPlainObject(value: object): boolean {
  const proto = Object.getPrototypeOf(value);
  return proto === null || proto === Object.prototype || Object.getPrototypeOf(proto) === null;
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]!);
  return btoa(binary);
}

/**
 * A JSON form of `value` that is the same in every process: object keys are
 * sorted, and sets, maps, dates, byte arrays and class instances give their
 * type and contents, so a set's order doesn't matter and a class instance is
 * kept apart from a plain object with the same fields.
 */
function canonical(value: unknown): unknown {
  if (value === null || typeof value !== 'object') {
    if (typeof value === 'bigint') return { __class__: 'BigInt', __value__: value.toString() };
    if (typeof value === 'function' || typeof value === 'symbol') return undefined;
    return value;
  }
  if (Array.isArray(value)) return value.map((item) => canonical(item) ?? null);
  const tag = Object.prototype.toString.call(value);
  if (tag === '[object Date]') return { __class__: 'Date', __value__: (value as Date).toISOString() };
  if (tag === '[object Uint8Array]') return { __class__: 'Uint8Array', __value__: bytesToBase64(value as Uint8Array) };
  if (tag === '[object Set]') {
    const items = [...(value as Set<unknown>)].map(canonical);
    return { __class__: 'Set', __value__: sortByJson(items) };
  }
  if (tag === '[object Map]') {
    const entries = [...(value as Map<unknown, unknown>)].map(([k, v]) => [canonical(k), canonical(v)]);
    return { __class__: 'Map', __value__: sortByJson(entries) };
  }
  const json = (value as { toJSON?: () => unknown }).toJSON;
  const source = typeof json === 'function' ? json.call(value) : value;
  if (source === null || typeof source !== 'object') return canonical(source);
  const sorted: Record<string, unknown> = {};
  for (const key of Object.keys(source).sort()) {
    const item = canonical((source as Record<string, unknown>)[key]);
    if (item !== undefined) sorted[key] = item;
  }
  if (isPlainObject(value)) return sorted;
  return { __class__: (value as object).constructor?.name ?? 'Object', __value__: sorted };
}

function sortByJson(items: unknown[]): unknown[] {
  return items
    .map((item) => [JSON.stringify(item), item] as const)
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([, item]) => item);
}

/** Builds a cache key from the activity name, arguments and runtime context. */
export function cacheKey(taskId: string, args: unknown[], context: unknown): string {
  let text: string;
  try {
    text = JSON.stringify(canonical([taskId, args, context ?? null]));
  } catch {
    text = String([taskId, args, context]);
  }
  return sha256Hex(text).slice(0, 32);
}
