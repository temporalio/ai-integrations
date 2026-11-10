import test from 'ava';
import type { LanguageModelV4CallOptions, LanguageModelV4GenerateResult } from '@ai-sdk/provider';
import { encodeCallOptionsFileData, encodeResultFileData } from '../file-data';

const bytes = new Uint8Array([137, 80, 78, 71, 0, 255]);
const base64 = Buffer.from(bytes).toString('base64');

test('call options without binary file data are returned unchanged', (t) => {
  const url = new URL('https://example.com/a.png');
  const options: LanguageModelV4CallOptions = {
    prompt: [
      { role: 'system', content: 'system' },
      {
        role: 'user',
        content: [
          { type: 'text', text: 'hi' },
          { type: 'file', mediaType: 'image/png', data: { type: 'data', data: base64 } },
          { type: 'file', mediaType: 'image/png', data: { type: 'url', url } },
        ],
      },
    ],
  };

  t.is(encodeCallOptionsFileData(options), options);
});

test('binary prompt file data is encoded as base64 without mutating the caller', (t) => {
  const data = { type: 'data', data: bytes } as const;
  const userFile = { type: 'file', mediaType: 'image/png', data } as const;
  const reasoningFile = { type: 'reasoning-file', mediaType: 'image/png', data } as const;
  const toolFile = { type: 'file', mediaType: 'image/png', data } as const;
  const options: LanguageModelV4CallOptions = {
    prompt: [
      { role: 'user', content: [{ type: 'text', text: 'hi' }, userFile] },
      { role: 'assistant', content: [reasoningFile] },
      {
        role: 'tool',
        content: [
          {
            type: 'tool-result',
            toolCallId: 'call-1',
            toolName: 'screenshot',
            output: { type: 'content', value: [{ type: 'text', text: 'shot' }, toolFile] },
          },
        ],
      },
    ],
  };

  const encoded = encodeCallOptionsFileData(options);

  t.deepEqual(encoded.prompt, [
    {
      role: 'user',
      content: [
        { type: 'text', text: 'hi' },
        { type: 'file', mediaType: 'image/png', data: { type: 'data', data: base64 } },
      ],
    },
    {
      role: 'assistant',
      content: [
        { type: 'reasoning-file', mediaType: 'image/png', data: { type: 'data', data: base64 } },
      ],
    },
    {
      role: 'tool',
      content: [
        {
          type: 'tool-result',
          toolCallId: 'call-1',
          toolName: 'screenshot',
          output: {
            type: 'content',
            value: [
              { type: 'text', text: 'shot' },
              { type: 'file', mediaType: 'image/png', data: { type: 'data', data: base64 } },
            ],
          },
        },
      ],
    },
  ]);
  t.is(data.data, bytes);
  t.is(userFile.data, data);
  t.is(reasoningFile.data, data);
  t.is(toolFile.data, data);
});

test('binary generated files in a model result are encoded as base64', (t) => {
  const url = { type: 'url', url: new URL('https://example.com/a.png') } as const;
  const result: LanguageModelV4GenerateResult = {
    content: [
      { type: 'text', text: 'done' },
      { type: 'file', mediaType: 'image/png', data: { type: 'data', data: bytes } },
      { type: 'reasoning-file', mediaType: 'image/png', data: { type: 'data', data: bytes } },
      { type: 'file', mediaType: 'image/png', data: url },
    ],
    finishReason: { unified: 'stop', raw: undefined },
    usage: {
      inputTokens: { total: 1, noCache: undefined, cacheRead: undefined, cacheWrite: undefined },
      outputTokens: { total: 1, text: undefined, reasoning: undefined },
    },
    warnings: [],
  };

  const encoded = encodeResultFileData(result);

  t.deepEqual(encoded.content, [
    { type: 'text', text: 'done' },
    { type: 'file', mediaType: 'image/png', data: { type: 'data', data: base64 } },
    { type: 'reasoning-file', mediaType: 'image/png', data: { type: 'data', data: base64 } },
    { type: 'file', mediaType: 'image/png', data: url },
  ]);
  t.is(encoded.content[3], result.content[3]);
  t.is(encodeResultFileData(encoded), encoded);
});
