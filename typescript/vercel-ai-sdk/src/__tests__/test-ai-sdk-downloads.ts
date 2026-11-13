import http from 'node:http';
import type { AddressInfo } from 'node:net';
import type { ExecutionContext } from 'ava';
import type { BaseContext } from '@temporalio/test-helpers';
import { helpers } from '@temporalio/test-helpers';
import { AiSdkPlugin } from '..';
import { downloadNotes } from './activities/ai-sdk';
import { localDownload, makeAiSdkTestFunction } from './helpers/ai-sdk';
import { RecordingProvider } from './stubs/ai-sdk';
import { experimentalDownloadWorkflow, fileUrlWorkflow, mediaTypeUrlWorkflow } from './workflows/ai-sdk';

const test = makeAiSdkTestFunction();

const pngBytes = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5E' +
    'rkJggg==',
  'base64'
);

async function serveFiles(t: { teardown(fn: () => Promise<void>): void }) {
  const requested: string[] = [];
  const server = http.createServer((req, res) => {
    requested.push(req.url ?? '');
    if (req.url === '/image') {
      res.writeHead(200, { 'content-type': 'image/png' }).end(pngBytes);
    } else if (req.url === '/notes') {
      res.writeHead(200, { 'content-type': 'text/plain; charset=utf-8' }).end('some notes');
    } else {
      res.writeHead(404).end();
    }
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  t.teardown(() => new Promise<void>((resolve) => server.close(() => resolve())));
  const baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  return { baseUrl, requested };
}

async function runFileUrlWorkflow(t: ExecutionContext<BaseContext>, provider: RecordingProvider) {
  const server = await serveFiles(t);
  const { createWorker, executeWorkflow } = helpers(t);
  const worker = await createWorker({
    plugins: [new AiSdkPlugin({ modelProvider: provider, download: localDownload })],
  });
  await worker.runUntil(async () => {
    const result = await executeWorkflow(fileUrlWorkflow, {
      args: [server.baseUrl],
      workflowExecutionTimeout: '10 seconds',
    });
    t.is(result, 'seen');
  });
  t.is(provider.calls.length, 1);
  const [message] = provider.calls[0]!.prompt;
  if (message?.role !== 'user') throw new Error('Expected a user message');
  return { ...server, content: message.content };
}

test('Prompt file URLs the model does not support are downloaded in the activity', async (t) => {
  const { content, requested } = await runFileUrlWorkflow(t, new RecordingProvider({}));

  t.deepEqual(content.slice(1), [
    {
      type: 'file',
      mediaType: 'image/png',
      data: { type: 'data', data: new Uint8Array(pngBytes) },
    },
    {
      type: 'file',
      mediaType: 'text/plain',
      data: { type: 'data', data: new TextEncoder().encode('some notes') },
    },
  ]);
  t.deepEqual(requested.sort(), ['/image', '/notes']);
});

test('Prompt file URLs the model supports reach the model as URL objects', async (t) => {
  const provider = new RecordingProvider({ 'image/*': [/^http:\/\/127\.0\.0\.1:/] });
  const { content, requested, baseUrl } = await runFileUrlWorkflow(t, provider);

  const [, image, notes] = content;
  if (image?.type !== 'file' || image.data.type !== 'url') throw new Error('Expected a URL part');
  t.true(image.data.url instanceof URL);
  t.is(image.data.url.href, `${baseUrl}/image`);
  t.is(image.mediaType, 'image');
  t.like(notes, { mediaType: 'text/plain', data: { type: 'data' } });
  t.deepEqual(requested, ['/notes']);
});

test('A downloaded image gets the media type that its bytes show', async (t) => {
  const provider = new RecordingProvider({});
  const server = await serveFiles(t);
  const { createWorker, executeWorkflow } = helpers(t);
  const worker = await createWorker({
    plugins: [new AiSdkPlugin({ modelProvider: provider, download: localDownload })],
  });
  await worker.runUntil(() =>
    executeWorkflow(mediaTypeUrlWorkflow, {
      args: [server.baseUrl],
      workflowExecutionTimeout: '10 seconds',
    })
  );

  const [message] = provider.calls[0]!.prompt;
  if (message?.role !== 'user') throw new Error('Expected a user message');
  t.like(message.content[1], { type: 'file', mediaType: 'image/png' });
});

test('The request body of a call that downloaded a file is not recorded', async (t) => {
  const provider = new RecordingProvider({}, { body: { file: 'the downloaded bytes' } });
  const server = await serveFiles(t);
  const { createWorker, startWorkflow } = helpers(t);
  const worker = await createWorker({
    plugins: [new AiSdkPlugin({ modelProvider: provider, download: localDownload })],
  });
  await worker.runUntil(async () => {
    const handle = await startWorkflow(fileUrlWorkflow, {
      args: [server.baseUrl],
      workflowExecutionTimeout: '10 seconds',
    });
    await handle.result();
    const { events } = await handle.fetchHistory();
    const results = events!.flatMap(
      (event) => event.activityTaskCompletedEventAttributes?.result?.payloads ?? []
    );
    t.is(results.length, 1);
    const recorded = JSON.parse(Buffer.from(results[0]!.data!).toString());
    t.false('body' in recorded.request);
  });
});

test('A new execution leaves the download of a URL to the model Activity', async (t) => {
  const provider = new RecordingProvider({});
  const server = await serveFiles(t);
  const { createWorker, startWorkflow } = helpers(t);
  const worker = await createWorker({
    plugins: [new AiSdkPlugin({ modelProvider: provider, download: localDownload })],
    activities: { downloadNotes },
  });
  await worker.runUntil(async () => {
    const handle = await startWorkflow(experimentalDownloadWorkflow, {
      args: [server.baseUrl],
      workflowExecutionTimeout: '10 seconds',
    });
    await handle.result();
    const { events } = await handle.fetchHistory();
    const scheduled = events!.flatMap(
      (event) => event.activityTaskScheduledEventAttributes?.activityType?.name ?? []
    );
    t.deepEqual(scheduled, ['invokeModel']);
  });
});
