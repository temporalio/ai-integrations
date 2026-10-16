/** Destination and credential-free replica routing shared across Temporal boundaries. */

import type { RunTreeConfig, WriteReplica } from 'langsmith/run_trees';
import { scrubSensitive } from './propagation';

type Destination = Pick<RunTreeConfig, 'project_name' | 'address'>;

export type ReplicaRouting = Pick<WriteReplica, 'projectName' | 'address' | 'primary' | 'updates' | 'reroot'>;

const SAFE_UPDATE_FIELDS = new Set(['tags', 'metadata', 'extra', 'reroot']);

/** Select one destination, clearing the other mode when inheriting an address. */
export function destination(preferred?: Destination, fallback?: Destination): Destination {
  const selected = preferred?.address || preferred?.project_name ? preferred : fallback;
  return {
    project_name: selected?.address ? undefined : selected?.project_name,
    address: selected?.address,
  };
}

function safeUpdates(value: unknown): unknown {
  if (value === null || typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return value;
  }
  if (Array.isArray(value)) return value.map(safeUpdates);
  if (value && typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype) {
    return Object.fromEntries(
      Object.entries(scrubSensitive(value as Record<string, unknown>) ?? {})
        .filter(([key]) => !['client', 'apiUrl', 'workspaceId'].includes(key))
        .map(([key, item]) => [key, safeUpdates(item)])
        .filter(([, item]) => item !== undefined)
    );
  }
  return undefined;
}

/** Follow LangSmith's header allowlist; clients, credentials and ingestion URLs stay local. */
export function replicaRouting(replicas: readonly WriteReplica[] | undefined): ReplicaRouting[] | undefined {
  return replicas?.map((replica) => ({
    ...(typeof replica.projectName === 'string' ? { projectName: replica.projectName } : {}),
    ...(typeof replica.address === 'string' ? { address: replica.address } : {}),
    ...(typeof replica.primary === 'boolean' ? { primary: replica.primary } : {}),
    ...(typeof replica.reroot === 'boolean' ? { reroot: replica.reroot } : {}),
    ...(replica.updates ? {
      updates: safeUpdates(Object.fromEntries(
        Object.entries(replica.updates).filter(([key]) => SAFE_UPDATE_FIELDS.has(key))
      )) as WriteReplica['updates'],
    } : {}),
  }));
}

/** The Temporal wire format uses Python's spelling; accept native JS headers too. */
export function replicaHeaders(replicas: readonly WriteReplica[]): unknown[] {
  return replicaRouting(replicas)!.map(({ projectName, ...replica }) => ({
    ...replica,
    ...(projectName === undefined ? {} : { project_name: projectName }),
  }));
}

export function normalizeReplicaHeaders(baggage: string): string {
  return baggage.split(',').map((item) => {
    const prefix = 'langsmith-replicas=';
    if (!item.startsWith(prefix)) return item;
    const replicas: unknown = JSON.parse(decodeURIComponent(item.slice(prefix.length)));
    if (!Array.isArray(replicas)) throw new TypeError('replica routing must be an array');
    const normalized = replicas.map((replica: unknown) => {
      if (Array.isArray(replica)) return { projectName: replica[0], updates: replica[1] } as WriteReplica;
      if (!replica || typeof replica !== 'object') throw new TypeError('replica routing must be an object');
      const record = replica as Record<string, unknown>;
      return { ...record, projectName: record.project_name ?? record.projectName } as WriteReplica;
    });
    return `${prefix}${encodeURIComponent(JSON.stringify(replicaRouting(normalized)))}`;
  }).join(',');
}

/** Restore a destination's credentials from the receiving process's own configuration. */
export function bindReplicaCredentials(
  replicas: readonly ReplicaRouting[] | undefined,
  localReplicas: readonly WriteReplica[] | undefined
): WriteReplica[] | undefined {
  if (replicas === undefined) return localReplicas ? [...localReplicas] : undefined;
  return replicas.map((replica) => {
    const local = localReplicas?.find(
      (candidate) => candidate.projectName === replica.projectName && candidate.address === replica.address
    );
    return {
      ...replica,
      apiKey: local?.apiKey,
      apiUrl: local?.apiUrl,
      workspaceId: local?.workspaceId,
      client: local?.client,
    };
  });
}
