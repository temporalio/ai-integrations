/**
 * Keeps the types of node and task values across the Activity boundary.
 *
 * Activity inputs and outputs travel through the Temporal payload converter as
 * JSON, which turns class instances into plain objects. The sender records
 * where values of known types sit in objects and arrays, by type ID, and puts
 * them in a JSON form; the receiver rebuilds them. Known types are LangChain
 * messages, `Uint8Array`, `Date`, and LangGraph's `Command` and `Send`.
 *
 * This module is part of the Workflow-sandbox import graph: it must not import
 * any Worker-only module.
 *
 * @module
 * @internal
 */

import { Command, Send, isCommand } from '@langchain/langgraph';
import { BaseMessage, mapChatMessagesToStoredMessages, mapStoredMessagesToChatMessages } from '@langchain/core/messages';
import type { StoredMessage } from '@langchain/core/messages';
import { ApplicationFailure } from '@temporalio/common';
import { UNKNOWN_VALUE_TYPE_FAILURE_TYPE } from './error-types';

/** A path into a value: object keys and array indices. */
export type ValuePath = (string | number)[];

/** `[path, typeId]` pairs recorded by {@link encodeValue}. */
export type ValueTypes = [ValuePath, string][];

export const COMMAND_TYPE_ID = 'langgraph.Command';
export const MESSAGE_TYPE_ID = 'langchain.BaseMessage';
export const BYTES_TYPE_ID = 'Uint8Array';
export const DATE_TYPE_ID = 'Date';

/** A `Command` in a form the payload converter keeps as is. */
export interface CommandPayload {
  graph: string | undefined;
  update: unknown;
  resume: unknown;
  goto: unknown;
}

/** A `Send` inside {@link CommandPayload.goto}. */
interface SendPayload {
  node: string;
  arg: unknown;
  timeout: unknown;
}

function isSend(value: unknown): value is Send {
  return value instanceof Send || (isPlainOrClassObject(value) && (value as { lg_name?: unknown }).lg_name === 'Send');
}

function isPlainOrClassObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (!isPlainOrClassObject(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === null || proto === Object.prototype || Object.getPrototypeOf(proto) === null;
}

function isUint8Array(value: unknown): value is Uint8Array {
  // `instanceof` fails across realms (a Worker-side value seen in the sandbox).
  return Object.prototype.toString.call(value) === '[object Uint8Array]';
}

function isDate(value: unknown): value is Date {
  return Object.prototype.toString.call(value) === '[object Date]';
}

/**
 * Converts a `Command` to a plain object. `Send`s in `goto` become
 * `{node, arg, timeout}` objects; node names stay strings.
 */
export function commandToPayload(command: Command): CommandPayload {
  const sendToPayload = (item: unknown): unknown =>
    isSend(item) ? ({ node: item.node, arg: item.args, timeout: item.timeout ?? null } satisfies SendPayload) : item;
  const goto = Array.isArray(command.goto) ? command.goto.map(sendToPayload) : sendToPayload(command.goto);
  return {
    graph: command.graph,
    update: command.update,
    resume: command.resume,
    goto,
  };
}

/**
 * Rebuilds a `Command` from {@link commandToPayload} output. Node names are
 * the only other values `goto` holds, so an object in `goto` is always a
 * serialized `Send`.
 */
export function commandFromPayload(payload: CommandPayload): Command {
  const payloadToSend = (item: unknown): unknown => {
    if (!isPlainOrClassObject(item)) return item;
    const { node, arg, timeout } = item as unknown as SendPayload;
    return new Send(node, arg, timeout != null ? { timeout: timeout as number } : undefined);
  };
  const goto = Array.isArray(payload.goto) ? payload.goto.map(payloadToSend) : payloadToSend(payload.goto);
  return new Command({
    graph: payload.graph ?? undefined,
    update: payload.update as Record<string, unknown> | undefined,
    resume: payload.resume,
    goto: (goto ?? undefined) as never,
  });
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]!);
  return btoa(binary);
}

function base64ToBytes(encoded: string): Uint8Array {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

/**
 * Records the known values in `value` under `path` in `valueTypes`, and returns
 * `value` with each of them in its JSON form. Unchanged subtrees are returned
 * as is.
 */
export function encodeValue(value: unknown, path: ValuePath, valueTypes: ValueTypes): unknown {
  if (isCommand(value)) {
    const payload = encodeValue(commandToPayload(value as Command), path, valueTypes);
    valueTypes.push([path, COMMAND_TYPE_ID]);
    return payload;
  }
  if (BaseMessage.isInstance(value)) {
    valueTypes.push([path, MESSAGE_TYPE_ID]);
    return mapChatMessagesToStoredMessages([value])[0];
  }
  if (isUint8Array(value)) {
    valueTypes.push([path, BYTES_TYPE_ID]);
    return bytesToBase64(value);
  }
  if (isDate(value)) {
    valueTypes.push([path, DATE_TYPE_ID]);
    return value.toISOString();
  }
  if (Array.isArray(value)) {
    let changed = false;
    const items = value.map((item, index) => {
      const encoded = encodeValue(item, [...path, index], valueTypes);
      if (encoded !== item) changed = true;
      return encoded;
    });
    return changed ? items : value;
  }
  if (isPlainObject(value)) {
    let changed = false;
    const entries = Object.entries(value).map(([key, item]) => {
      const encoded = encodeValue(item, [...path, key], valueTypes);
      if (encoded !== item) changed = true;
      return [key, encoded] as const;
    });
    return changed ? Object.fromEntries(entries) : value;
  }
  return value;
}

const TYPE_ID = Symbol('typeId');

interface TypeTree {
  [TYPE_ID]?: string;
  children: Map<string, TypeTree>;
}

/** Rebuilds the values {@link encodeValue} recorded, leaving `value` itself unchanged. */
export function restoreValue(value: unknown, valueTypes: ValueTypes | null | undefined): unknown {
  if (!valueTypes || valueTypes.length === 0) return value;
  const root: TypeTree = { children: new Map() };
  for (const [path, typeId] of valueTypes) {
    let node = root;
    for (const key of path) {
      const name = String(key);
      let child = node.children.get(name);
      if (child === undefined) {
        child = { children: new Map() };
        node.children.set(name, child);
      }
      node = child;
    }
    node[TYPE_ID] = typeId;
  }
  return rebuild(value, root);
}

function rebuild(value: unknown, node: TypeTree): unknown {
  let current = value;
  if (node.children.size > 0 && isPlainOrClassObject(current)) {
    const copy: Record<string, unknown> | unknown[] = Array.isArray(current) ? [...current] : { ...current };
    for (const [key, child] of node.children) {
      const container = copy as Record<string, unknown>;
      if (key in container) container[key] = rebuild(container[key], child);
    }
    current = copy;
  }
  const typeId = node[TYPE_ID];
  return typeId === undefined ? current : rebuildTyped(typeId, current);
}

function rebuildTyped(typeId: string, value: unknown): unknown {
  switch (typeId) {
    case COMMAND_TYPE_ID:
      return commandFromPayload(value as CommandPayload);
    case MESSAGE_TYPE_ID:
      return mapStoredMessagesToChatMessages([value as StoredMessage])[0];
    case BYTES_TYPE_ID:
      return base64ToBytes(value as string);
    case DATE_TYPE_ID:
      return new Date(value as string);
    default:
      throw ApplicationFailure.nonRetryable(
        `Unknown value type ID '${typeId}': this Worker can't rebuild it. All Workers must run compatible ` +
          'versions of @temporalio/langgraph.',
        UNKNOWN_VALUE_TYPE_FAILURE_TYPE
      );
  }
}
