import { BedrockModel, type McpClient, type Model } from '@strands-agents/sdk';
import { ModelActivity } from './model-activity';
import type { InvokeModelInput, InvokeModelStreamingInput } from './model-activity';
import {
  buildCallToolActivity,
  buildListToolsActivity,
  callToolActivityName,
  listToolsActivityName,
} from './temporal-mcp-client';

/** The models one {@link StrandsPlugin} serves. */
export interface ModelRegistration {
  /** Whether the plugin was given `models`; if not, it serves Strands' default `BedrockModel`. */
  supplied: boolean;
  factories: Record<string, () => Model>;
  /** The model a `TemporalAgent` without `model` gets: the only one in `factories`, if exactly one. */
  defaultName?: string;
}

/** An MCP server one {@link StrandsPlugin} serves. */
export interface McpRegistration {
  factory: () => McpClient;
  /** The effective idle timeout: `mcpConnectionIdleTimeout` or its default, in milliseconds. */
  idleMs: number;
}

type Registration = { kind: 'model'; models: ModelRegistration } | { kind: 'mcp'; server: McpRegistration };

type Activities = Record<string, (...args: never[]) => unknown>;

// What each Activity function a StrandsPlugin built serves. The SDK's plugin merge keeps the last
// Activity of each name, so a StrandsPlugin checks the Activities already registered before it
// adds its own: it merges with another StrandsPlugin's models and refuses anything else.
const registrations = new WeakMap<object, Registration>();

const MODEL_ACTIVITIES = ['invokeModel', 'invokeModelStreaming'];
const defaultBedrock = (): Model => new BedrockModel({});

/**
 * The Activities to register for one plugin, given the Activities already registered: the
 * `existing` ones plus this plugin's, with the model Activities serving the models of every
 * StrandsPlugin on the Worker. Throws a `TypeError` instead of replacing a different Activity.
 */
export function addStrandsActivities(
  existing: Activities,
  models: ModelRegistration,
  mcpServers: Record<string, McpRegistration>
): Activities {
  const result: Activities = { ...existing };
  let merged = models;
  for (const name of MODEL_ACTIVITIES) {
    const other = registrationOf(existing, name);
    if (other?.kind !== 'model') continue;
    merged = mergeModels(other.models, merged);
  }
  Object.assign(result, buildModelActivities(merged));
  for (const [server, entry] of Object.entries(mcpServers)) {
    const names = [listToolsActivityName(server), callToolActivityName(server)];
    const others = names.map((name) => registrationOf(existing, name));
    if (others.some((other) => other?.kind === 'mcp')) {
      for (const other of others) {
        if (other?.kind !== 'mcp' || other.server.factory !== entry.factory || other.server.idleMs !== entry.idleMs) {
          throw new TypeError(
            `Duplicate registration of MCP server '${server}': two StrandsPlugin instances on this Worker ` +
              'register it with different factories or idle timeouts. Register each MCP server once.'
          );
        }
      }
      continue; // The same server, from the same plugin options: keep the existing Activities.
    }
    const list = buildListToolsActivity(server, entry.factory, entry.idleMs);
    const call = buildCallToolActivity(server, entry.factory, entry.idleMs);
    registrations.set(list, { kind: 'mcp', server: entry });
    registrations.set(call, { kind: 'mcp', server: entry });
    Object.assign(result, { [names[0]!]: list, [names[1]!]: call });
  }
  return result;
}

/** The registration of the Activity `name` in `existing`; throws if a non-Strands Activity has it. */
function registrationOf(existing: Activities, name: string): Registration | undefined {
  if (!Object.prototype.hasOwnProperty.call(existing, name) || typeof existing[name] !== 'function') {
    return undefined;
  }
  const registration = registrations.get(existing[name]!);
  if (registration === undefined) {
    throw new TypeError(
      `Duplicate registration of Activity '${name}': StrandsPlugin would replace a different Activity with this name. ` +
        'Rename or remove the other registration.'
    );
  }
  return registration;
}

function mergeModels(a: ModelRegistration, b: ModelRegistration): ModelRegistration {
  if (a === b || !b.supplied) return a;
  if (!a.supplied) return b;
  // A Map, so that names like `constructor` or `__proto__` are plain keys.
  const factories = new Map(Object.entries(a.factories));
  for (const [name, factory] of Object.entries(b.factories)) {
    if (factories.has(name) && factories.get(name) !== factory) {
      throw new TypeError(
        `Duplicate registration of model '${name}': two StrandsPlugin instances on this Worker ` +
          'register different factories for it. Register each model once.'
      );
    }
    factories.set(name, factory);
  }
  if (a.defaultName !== undefined && b.defaultName !== undefined && a.defaultName !== b.defaultName) {
    throw new TypeError(
      `Conflicting default models '${a.defaultName}' and '${b.defaultName}': two StrandsPlugin instances on this ` +
        'Worker each register a single model, so a TemporalAgent without `model` has no single default. ' +
        'Register the models in one StrandsPlugin.'
    );
  }
  return { supplied: true, factories: Object.fromEntries(factories), defaultName: a.defaultName ?? b.defaultName };
}

function buildModelActivities(models: ModelRegistration): Activities {
  const modelActivity = models.supplied
    ? new ModelActivity(models.factories, models.defaultName)
    : new ModelActivity({ bedrock: defaultBedrock }, 'bedrock');
  const activities: Activities = {
    invokeModel: (input: InvokeModelInput) => modelActivity.invokeModel(input),
    invokeModelStreaming: (input: InvokeModelStreamingInput) => modelActivity.invokeModelStreaming(input),
  };
  for (const fn of Object.values(activities)) registrations.set(fn, { kind: 'model', models });
  return activities;
}
