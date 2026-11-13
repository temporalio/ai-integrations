import { defaultPayloadConverter } from '@temporalio/common';

/** The first field that the payload converter changed, as `$.a.b[0]`. */
export interface PayloadChange {
  path: string;
  sent: string;
  received: string;
}

/**
 * Converts `value` to a payload and back with the SDK's default payload converter.
 * Returns the first field that did not come back deeply equal, or `undefined`.
 */
export function findPayloadChange(value: unknown): PayloadChange | undefined {
  const received = defaultPayloadConverter.fromPayload(defaultPayloadConverter.toPayload(value));
  return firstChange(value, received, '$');
}

function firstChange(sent: unknown, received: unknown, path: string): PayloadChange | undefined {
  if (Object.is(sent, received)) return undefined;
  const change = { path, sent: describe(sent), received: describe(received) };
  if (typeof sent !== 'object' || sent === null) return change;
  if (typeof received !== 'object' || received === null) return change;
  if (Object.getPrototypeOf(sent) !== Object.getPrototypeOf(received)) return change;
  if (ArrayBuffer.isView(sent)) {
    const a = bytes(sent);
    const b = bytes(received as ArrayBufferView);
    return a.length === b.length && a.every((byte, i) => byte === b[i]) ? undefined : change;
  }
  if (Array.isArray(sent)) {
    const other = received as unknown[];
    if (sent.length !== other.length) return change;
    for (let i = 0; i < sent.length; i++) {
      const inner = firstChange(sent[i], other[i], `${path}[${i}]`);
      if (inner !== undefined) return inner;
    }
    return undefined;
  }
  const sentRecord = sent as Record<string, unknown>;
  const receivedRecord = received as Record<string, unknown>;
  // JSON leaves out `undefined` fields, and a missing field reads back as `undefined` too.
  for (const key of new Set([...Object.keys(sentRecord), ...Object.keys(receivedRecord)])) {
    const step = /^[A-Za-z_$][\w$]*$/.test(key) ? `.${key}` : `[${JSON.stringify(key)}]`;
    const inner = firstChange(sentRecord[key], receivedRecord[key], path + step);
    if (inner !== undefined) return inner;
  }
  return undefined;
}

function bytes(view: ArrayBufferView): Uint8Array {
  return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
}

function describe(value: unknown): string {
  if (typeof value === 'string') return JSON.stringify(value);
  if (typeof value !== 'object' || value === null) return String(value);
  return Object.getPrototypeOf(value)?.constructor?.name ?? 'Object';
}
