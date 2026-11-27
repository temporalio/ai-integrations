import type { ExecutionContext, TestFn } from 'ava';
import type { BaseContext } from '@temporalio/test-helpers';
import { test as anyTest, createTestWorkflowBundle } from '@temporalio/test-helpers';
import type { BundlerPlugin } from '@temporalio/worker';
import { createTestEnvironment } from './environment';

export interface TestFunctionOptions {
  workflowsPath: string;
  workflowInterceptorModules?: string[];
  plugins?: BundlerPlugin[];
}

/**
 * Create an ava TestFn whose `before` hook sets up `t.context.env` and
 * `t.context.workflowBundle`.
 */
export function makeTestFunction(opts: TestFunctionOptions): TestFn<BaseContext> {
  const test = anyTest as TestFn<BaseContext>;
  test.before(async (t: ExecutionContext<BaseContext>) => {
    const env = await createTestEnvironment();
    const workflowBundle = await createTestWorkflowBundle({
      workflowsPath: opts.workflowsPath,
      workflowInterceptorModules: opts.workflowInterceptorModules,
      plugins: opts.plugins,
    });
    t.context = { env, workflowBundle };
  });
  test.after.always(async (t: ExecutionContext<BaseContext>) => {
    if (t.context.env) {
      await t.context.env.teardown();
    }
  });
  return test;
}
