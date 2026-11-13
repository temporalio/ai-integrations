import type {
  LanguageModelV4CallOptions,
  LanguageModelV4Content,
  LanguageModelV4GenerateResult,
  LanguageModelV4Message,
  LanguageModelV4StreamPart,
  SharedV4FileData,
} from '@ai-sdk/provider';
import { fromByteArray } from 'base64-js';

// Temporal's JSON payload converter turns a `Uint8Array` into a plain object. The v4 spec
// accepts a base64 string wherever it accepts bytes, so file bytes cross the boundary as base64.
//
// No patch gate is necessary. Only Activity payloads change, and replay does not compare
// them. A gate would also keep the corrupted bytes for executions that are in flight.

type PromptPart = Exclude<LanguageModelV4Message, { role: 'system' }>['content'][number];

// Keeps the original array when no item changed, so inputs without bytes keep their exact payload.
function mapChanged<T>(items: T[], encode: (item: T) => T): T[] {
  const next = items.map(encode);
  return next.some((item, i) => item !== items[i]) ? next : items;
}

function encodeFilePart<T extends { data: SharedV4FileData }>(part: T): T {
  const { data } = part;
  // `isView` instead of `instanceof` because Workflow sandbox and host realms differ.
  if (data.type !== 'data' || !ArrayBuffer.isView(data.data)) return part;
  return { ...part, data: { ...data, data: fromByteArray(data.data as Uint8Array) } };
}

function encodePromptPart(part: PromptPart): PromptPart {
  if (part.type === 'file' || part.type === 'reasoning-file') return encodeFilePart(part);
  if (part.type !== 'tool-result' || part.output.type !== 'content') return part;
  const value = mapChanged(part.output.value, (item) =>
    item.type === 'file' ? encodeFilePart(item) : item
  );
  return value === part.output.value ? part : { ...part, output: { ...part.output, value } };
}

export function encodeCallOptionsFileData(
  options: LanguageModelV4CallOptions
): LanguageModelV4CallOptions {
  const prompt = mapChanged(options.prompt, (message) => {
    if (message.role === 'system') return message;
    const content = mapChanged<PromptPart>(message.content, encodePromptPart);
    if (content === message.content) return message;
    return { ...message, content } as LanguageModelV4Message;
  });
  return prompt === options.prompt ? options : { ...options, prompt };
}

export function encodeContentFileData<T extends LanguageModelV4Content | LanguageModelV4StreamPart>(
  part: T
): T {
  if (part.type !== 'file' && part.type !== 'reasoning-file') return part;
  return encodeFilePart(part) as T;
}

export function encodeResultFileData(
  result: LanguageModelV4GenerateResult
): LanguageModelV4GenerateResult {
  const content = mapChanged(result.content, encodeContentFileData);
  return content === result.content ? result : { ...result, content };
}
