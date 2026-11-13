import type { LanguageModelV4Prompt } from '@ai-sdk/provider';
import { TemporalEmbeddingModel, TemporalLanguageModel } from '../../workflow';

const file = {
  type: 'file',
  mediaType: 'image/png',
  data: { type: 'data', data: new Uint8Array([137, 80, 78, 71, 0, 255]) },
} as const;

const urlFile = {
  type: 'file',
  mediaType: 'image/png',
  data: { type: 'url', url: new URL('https://example.test/a.png') },
} as const;

// Calls the Temporal models as `generateText` and `embedMany` would, with every non-JSON
// value the call options allow.
export async function payloadContractWorkflow(withUrl: boolean): Promise<string> {
  const prompt: LanguageModelV4Prompt = [
    { role: 'user', content: [{ type: 'text', text: 'hi' }, file, ...(withUrl ? [urlFile] : [])] },
    { role: 'assistant', content: [{ ...file, type: 'reasoning-file' }] },
    {
      role: 'tool',
      content: [
        {
          type: 'tool-result',
          toolCallId: 'call-1',
          toolName: 'screenshot',
          output: { type: 'content', value: [file] },
        },
      ],
    },
  ];
  const abortSignal = new AbortController().signal;
  try {
    const result = await new TemporalLanguageModel('m').doGenerate({ prompt, abortSignal });
    await new TemporalEmbeddingModel('e').doEmbed({ values: ['a'], abortSignal });
    return `timestamp is Date: ${result.response?.timestamp instanceof Date}`;
  } catch (err) {
    return `error: ${(err as Error).message}`;
  }
}
