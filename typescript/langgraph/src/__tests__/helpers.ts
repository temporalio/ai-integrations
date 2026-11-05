 
/**
 * Shared test scaffolding: a local Temporal server per test file, a Worker
 * factory with one cached Workflow bundle, and history helpers.
 */

import path from 'node:path';
import type { TestFn } from 'ava';
import { TestWorkflowEnvironment } from '@temporalio/testing';
import { DefaultLogger, Runtime, bundleWorkflowCode, Worker } from '@temporalio/worker';
import type { WorkflowBundle } from '@temporalio/worker';
import { LangGraphPlugin } from '../index';
import { defineLangGraph } from '../workflow';
import { startLocalWithRetry } from './helpers/environment';

function isSet(env: string | undefined, def: boolean): boolean {
  if (env === undefined) return def;
  const value = env.toLowerCase();
  return value === '1' || value === 't' || value === 'true';
}

Runtime.install({ logger: new DefaultLogger('WARN') });

/** `REUSE_V8_CONTEXT=false` runs Workers with a new V8 context per Workflow. */
export const REUSE_V8_CONTEXT = isSet(process.env.REUSE_V8_CONTEXT, true);

/**
 * The compiled test Workflows. The bundle imports the plugin's compiled
 * modules, so the Workflows and the plugin's interceptor modules share them.
 */
export const workflowsPath = path.resolve(__dirname, './workflows.js');

/** A unique task queue or Workflow ID. */
export function uid(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.floor(Math.random() * 1e9)}`;
}

/** Starts a local Temporal server before the file's tests and stops it after them. */
export function setupTestEnv(test: TestFn): () => TestWorkflowEnvironment {
  let env: TestWorkflowEnvironment | undefined;
  test.before(async () => {
    env = await startLocalWithRetry(() => TestWorkflowEnvironment.createLocal());
  });
  test.after.always(async () => {
    await env?.teardown();
  });
  return () => env!;
}

let bundle: Promise<WorkflowBundle> | undefined;

/**
 * The test Workflow bundle, built once per process. The plugin's bundler
 * configuration doesn't depend on its definitions.
 */
export function workflowBundle(): Promise<WorkflowBundle> {
  bundle ??= bundleWorkflowCode({
    workflowsPath,
    plugins: [new LangGraphPlugin(defineLangGraph())],
    logger: { trace() {}, debug() {}, info() {}, warn() {}, error: console.error, log() {} } as never,
  });
  return bundle;
}

export interface WithWorkerOptions {
  taskQueue: string;
  plugin: LangGraphPlugin;
  /**
   * Defaults to 0, so every Workflow task replays the history from the start,
   * as after a Worker restart or a cache eviction.
   */
  maxCachedWorkflows?: number;
}

/** The plugin of each task queue's Worker, which {@link execute} replays with. */
const plugins = new Map<string, LangGraphPlugin>();

/** Runs `fn` while a Worker with `plugin` polls `taskQueue`. */
export async function withWorker<T>(
  env: TestWorkflowEnvironment,
  options: WithWorkerOptions,
  fn: () => Promise<T>
): Promise<T> {
  const worker = await Worker.create({
    connection: env.nativeConnection,
    namespace: env.namespace ?? 'default',
    taskQueue: options.taskQueue,
    workflowBundle: await workflowBundle(),
    reuseV8Context: REUSE_V8_CONTEXT,
    plugins: [options.plugin],
    maxCachedWorkflows: options.maxCachedWorkflows ?? 0,
  });
  plugins.set(options.taskQueue, options.plugin);
  return worker.runUntil(fn());
}

/** Replays a completed Workflow's history with `plugin`, failing on nondeterminism. */
export async function replay(env: TestWorkflowEnvironment, plugin: LangGraphPlugin, workflowId: string): Promise<void> {
  const history = await env.client.workflow.getHandle(workflowId).fetchHistory();
  await Worker.runReplayHistory(
    { workflowBundle: await workflowBundle(), reuseV8Context: REUSE_V8_CONTEXT, plugins: [plugin] },
    history,
    workflowId
  );
}

/** A history event, narrowed to the fields the tests read. */
export interface HistoryEvent {
  activityTaskScheduledEventAttributes?: { activityType?: { name?: string | null } | null } | null;
  userMetadata?: { summary?: { data?: Uint8Array | null } | null } | null;
}

/** The decoded summaries of the Activities a history schedules, in order. */
export function activitySummaries(events: HistoryEvent[]): string[] {
  return events
    .filter((event) => event.activityTaskScheduledEventAttributes)
    .map((event) => {
      const data = event.userMetadata?.summary?.data;
      return data ? Buffer.from(data).toString('utf8') : '';
    });
}

/** The names of the Activities a history schedules, in order. */
export function scheduledActivities(events: HistoryEvent[]): string[] {
  return events
    .filter((event) => event.activityTaskScheduledEventAttributes)
    .map((event) => event.activityTaskScheduledEventAttributes!.activityType?.name ?? '');
}

/**
 * Executes a Workflow, failing it after 30 seconds, then replays its history
 * with the task queue's plugin, and returns its result or rethrows its failure.
 */
export async function execute<T>(
  env: TestWorkflowEnvironment,
  workflow: string | ((...args: any[]) => Promise<T>),
  options: { taskQueue: string; args: unknown[]; workflowId?: string }
): Promise<T> {
  const workflowId = options.workflowId ?? uid('test-workflow');
  const result = env.client.workflow.execute(workflow as never, {
    taskQueue: options.taskQueue,
    workflowId,
    args: options.args as never,
    workflowExecutionTimeout: '30s',
  }) as Promise<T>;
  await result.catch(() => undefined);
  const plugin = plugins.get(options.taskQueue);
  if (plugin !== undefined) await replay(env, plugin, workflowId);
  return result;
}
