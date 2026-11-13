import type { TestFn } from 'ava';
import type { BaseContext } from '@temporalio/test-helpers';
import { test as anyTest, createBaseBundlerOptions, TestWorkflowEnvironment } from '@temporalio/test-helpers';
import { workflowInterceptorModules } from '@temporalio/testing';
import { bundleWorkflowCode, DefaultLogger } from '@temporalio/worker';
import { AiSdkPlugin } from '../..';
import { helloWorkflowGenerator, TestProvider } from '../stubs/ai-sdk';
import { startLocalWithRetry } from './environment';

export const remoteTests = ['1', 't', 'true'].includes((process.env.AI_SDK_REMOTE_TESTS ?? 'false').toLowerCase());

/** Register the local server and Workflow bundle for one feature suite. */
export function makeAiSdkTestFunction(): TestFn<BaseContext> {
  // AVA runs one file at a time; small feature suites bound concurrent Workers.
  const test = anyTest as TestFn<BaseContext>;
  test.before(async (t) => {
    const env = await startLocalWithRetry(() => TestWorkflowEnvironment.createLocal());
    const workflowBundle = await bundleWorkflowCode({
      ...createBaseBundlerOptions(),
      workflowsPath: require.resolve('../workflows/ai-sdk'),
      workflowInterceptorModules,
      logger: new DefaultLogger('WARN'),
      // The plugin's configureBundler prepends the polyfill installer to the bundle entry;
      // the test workflows import `ai` before the workflow entry point and rely on it.
      plugins: [new AiSdkPlugin({ modelProvider: new TestProvider(helloWorkflowGenerator()) })],
    });
    t.context = { env, workflowBundle };
  });

  test.after.always(async (t) => {
    await t.context.env?.teardown();
  });
  return test;
}

// The default `createDownload()` refuses private addresses such as 127.0.0.1.
export async function localDownload({
  url,
  abortSignal,
}: {
  url: URL;
  abortSignal?: AbortSignal;
}): Promise<{ data: Uint8Array; mediaType: string | undefined }> {
  const response = await fetch(url, { signal: abortSignal });
  const data = new Uint8Array(await response.arrayBuffer());
  return { data, mediaType: response.headers.get('content-type') ?? undefined };
}
