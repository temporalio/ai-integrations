/**
 * Server-free unit tests that assert the shape of `configureBundler` /
 * `configureWorker` output without bundling or executing a Workflow.
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import test from 'ava';
import { START, StateGraph, task } from '@langchain/langgraph';
import type { BundleOptions, WorkerOptions } from '@temporalio/worker';
import { createActivities } from '../activities';
import { internalsOf } from '../definitions';
import { LangGraphPlugin, defineLangGraph } from '../index';
import { interceptors as polyfillInterceptors } from '../load-polyfills';
import stackTraceLimitLoader from '../stack-trace-limit-loader';
import { ValueState } from './graphs';

interface WebpackConfigLike {
  plugins?: { name?: string; apply?: (compiler: unknown) => void }[];
  module?: { rules?: { test?: RegExp; use?: { loader: string }[] }[] };
}

test('configureBundler brackets the user modules with the polyfill loader and the run interceptors', (t) => {
  const plugin = new LangGraphPlugin(defineLangGraph());
  const { workflowInterceptorModules } = plugin.configureBundler({
    workflowsPath: 'wf',
    workflowInterceptorModules: ['user-interceptors'],
  } as BundleOptions);
  // The polyfill loader must be first so the globals install before any other
  // per-Workflow module (interceptors, then the user's Workflows) loads LangChain.
  t.deepEqual(workflowInterceptorModules, [
    require.resolve('../load-polyfills'),
    'user-interceptors',
    require.resolve('../workflow-interceptors'),
  ]);
  t.deepEqual(polyfillInterceptors(), {});
});

test('configureBundler appends the async_hooks replacement and the LangGraph loader, preserving a user hook', (t) => {
  const userHook = (config: WebpackConfigLike): WebpackConfigLike => {
    config.plugins = [...(config.plugins ?? []), { name: 'user-plugin' }];
    config.module = { rules: [{ test: /\.user$/ }] };
    return config;
  };
  const plugin = new LangGraphPlugin(defineLangGraph());
  const { webpackConfigHook } = plugin.configureBundler({
    workflowsPath: 'wf',
    webpackConfigHook: userHook,
  } as unknown as BundleOptions);
  const result = webpackConfigHook!({ plugins: [] } as never) as WebpackConfigLike;

  const [user, replaceAsyncHooks] = result.plugins ?? [];
  t.is(user?.name, 'user-plugin');
  t.is(result.plugins?.length, 2);
  const replacements: { pattern: RegExp; replace: (resource: { request: string }) => void }[] = [];
  replaceAsyncHooks!.apply!({
    webpack: {
      NormalModuleReplacementPlugin: class {
        constructor(pattern: RegExp, replace: (resource: { request: string }) => void) {
          replacements.push({ pattern, replace });
        }
        apply() {}
      },
    },
  });
  const [{ pattern, replace }] = replacements as [(typeof replacements)[number]];
  for (const request of ['async_hooks', 'node:async_hooks']) {
    t.true(pattern.test(request), request);
    const resource = { request };
    replace(resource);
    t.is(resource.request, require.resolve('../async-hooks-shim'));
  }
  t.false(pattern.test('async_hooks/x'));

  const [userRule, loaderRule] = result.module?.rules ?? [];
  t.deepEqual(userRule, { test: /\.user$/ });
  t.deepEqual(loaderRule?.use, [{ loader: require.resolve('../stack-trace-limit-loader') }]);
  for (const file of [
    '/app/node_modules/@langchain/langgraph/dist/errors.js',
    'C:\\app\\node_modules\\@langchain\\langgraph\\dist\\pregel\\index.cjs',
  ]) {
    t.true(loaderRule!.test!.test(file), file);
  }
  t.false(loaderRule!.test!.test('/app/node_modules/@langchain/core/dist/errors/index.js'));
});

test("the loader drops LangGraph's Error.stackTraceLimit assignments", (t) => {
  const errors = readFileSync(path.join(path.dirname(require.resolve('@langchain/langgraph')), 'errors.js'), 'utf8');
  t.regex(errors, /Error\.stackTraceLimit = 0;/);
  const loaded = stackTraceLimitLoader(errors);
  t.notRegex(loaded, /Error\.stackTraceLimit\s*=/);
  t.regex(loaded, /const prevLimit = Error\.stackTraceLimit;/);
});

const base = { taskQueue: 'tq' } as WorkerOptions;

const duplicate = (name: string) => ({
  instanceOf: TypeError,
  message: new RegExp(`^Duplicate registration of Activity '${name}': LangGraphPlugin would replace`),
});

function definitionsWith(node: (state: typeof ValueState.State) => { value: string }, streamingTopic?: string) {
  const graph = new StateGraph(ValueState)
    .addNode('node', node, { metadata: { executeIn: 'activity' } })
    .addNode('other', node, { metadata: { executeIn: 'activity' } })
    .addEdge(START, 'node')
    .addEdge('node', 'other');
  return defineLangGraph({ graphs: { 'my-graph': graph }, streamingTopic });
}

const node = (state: typeof ValueState.State) => ({ value: state.value + 'a' });

test('configureWorker refuses to replace a different Activity with the same name', (t) => {
  // A caller Activity, or another plugin's node function or streaming options
  // under the same graph and node names, is refused.
  const callerActivities = { ...base, activities: { 'my-graph.node': async () => 'caller' } };
  t.throws(() => new LangGraphPlugin(definitionsWith(node)).configureWorker(callerActivities), duplicate('my-graph.node'));
  const registered = new LangGraphPlugin(definitionsWith(node)).configureWorker(base);
  const otherNode = new LangGraphPlugin(definitionsWith((state) => ({ value: state.value + 'b' })));
  t.throws(() => otherNode.configureWorker(registered), duplicate('my-graph.node'));
  const otherTopic = new LangGraphPlugin(definitionsWith(node, 'tokens'));
  t.throws(() => otherTopic.configureWorker(registered), duplicate('my-graph.node'));
  // Another node's Activity registered under this name, though it runs the same function.
  const activities = createActivities(internalsOf(definitionsWith(node)));
  const aliased = { ...base, activities: { 'my-graph.node': activities['my-graph.other'] } };
  t.throws(() => new LangGraphPlugin(definitionsWith(node)).configureWorker(aliased), duplicate('my-graph.node'));
  // A task is refused the same way.
  const first = task('my_task', (x: number) => x + 1);
  const second = task('my_task', (x: number) => x + 2);
  const taskOptions = { activityOptions: { my_task: { executeIn: 'activity' as const } } };
  const withTask = new LangGraphPlugin(defineLangGraph({ tasks: [first], ...taskOptions })).configureWorker(base);
  t.throws(
    () => new LangGraphPlugin(defineLangGraph({ tasks: [second], ...taskOptions })).configureWorker(withTask),
    duplicate('my_task')
  );

  // configureWorker accepts the same instance twice (NativeConnection and
  // Worker), the same definitions twice, definitions of the same graph, and
  // createActivities for the same definitions.
  const definitions = definitionsWith(node);
  const plugin = new LangGraphPlugin(definitions);
  t.notThrows(() => plugin.configureWorker(plugin.configureWorker(base)));
  t.notThrows(() => new LangGraphPlugin(definitions).configureWorker(plugin.configureWorker(base)));
  t.notThrows(() => new LangGraphPlugin(definitionsWith(node)).configureWorker(registered));
  const callerSet = { ...base, activities: createActivities(internalsOf(definitions)) };
  t.notThrows(() => new LangGraphPlugin(definitions).configureWorker(callerSet));
  t.notThrows(() =>
    new LangGraphPlugin(defineLangGraph({ tasks: [first], ...taskOptions })).configureWorker(withTask)
  );
});
