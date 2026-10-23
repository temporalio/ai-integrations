/**
 * Server-free unit tests that assert the shape of `configureBundler` /
 * `configureWorker` output without bundling or executing a Workflow.
 */

import { createRequire } from 'node:module';
import path from 'node:path';

import test from 'ava';
import type { LlmResponse } from '@google/adk';
import { MockActivityEnvironment } from '@temporalio/testing';
import type { BundleOptions, WorkerOptions } from '@temporalio/worker';

import { GoogleAdkPlugin, type GoogleAdkPluginOptions } from '../index';
import { interceptors as autoRouteInterceptors } from '../auto-route';
import { interceptors as polyfillInterceptors } from '../load-polyfills';
import type { InvokeModelArgs, ModelActivities } from '../model';
import { fakeModelProvider, mockMCPToolset } from '../testing';
import { sandboxCompatResolveHook } from './helpers';

interface NamedPlugin {
  name?: string;
}
interface WebpackConfigLike {
  plugins?: NamedPlugin[];
}

test('configureBundler stubs ADK node-only packages and disallowed builtins', (t) => {
  const plugin = new GoogleAdkPlugin();
  const { ignoreModules } = plugin.configureBundler({ workflowsPath: 'wf' } as BundleOptions);
  const ignored = new Set(ignoreModules ?? []);

  for (const pkg of [
    'google-auth-library',
    'googleapis',
    '@modelcontextprotocol/sdk',
    '@google-cloud/storage',
    'express',
    // ADK 2.0's `skills/loader.js` (on the web surface) imports the zip reader.
    'adm-zip',
  ]) {
    t.true(ignored.has(pkg), `expected ADK node-only package ${pkg} to be stubbed`);
  }
  // `dns/promises` is what ADK >= 1.5.0's `load_web_page.js` imports for its
  // (function-body-only) host resolution. `net` from the same file is
  // redirected to a shim at resolve time before this alias could apply; it
  // stays listed because the disallowed-builtins list is derived wholesale
  // from `builtinModules`.
  for (const builtin of ['fs', 'child_process', 'dns/promises', 'net']) {
    t.true(ignored.has(builtin), `expected disallowed builtin ${builtin} to be stubbed`);
  }
  // The three sandbox-polyfilled builtins must stay resolvable.
  for (const polyfilled of ['assert', 'url', 'util']) {
    t.false(ignored.has(polyfilled), `expected polyfilled builtin ${polyfilled} to remain`);
  }
  // The pure-JS OpenTelemetry tracing packages must stay resolvable:
  // `@temporalio/interceptors-opentelemetry` constructs a `BasicTracerProvider`
  // from them inside the sandbox — the SDK's only replay-safe workflow span
  // path. Re-stubbing any of these breaks composing with `OpenTelemetryPlugin`.
  for (const otel of ['@opentelemetry/api', '@opentelemetry/sdk-trace-base', '@opentelemetry/resources']) {
    t.false(ignored.has(otel), `expected OpenTelemetry package ${otel} to remain resolvable`);
  }
});

test('configureBundler brackets the user modules with the polyfill loader, auto-route and failure interceptor', (t) => {
  const plugin = new GoogleAdkPlugin();
  const { workflowInterceptorModules } = plugin.configureBundler({
    workflowsPath: 'wf',
    workflowInterceptorModules: ['user-interceptors'],
  } as BundleOptions);
  // The polyfill loader must be first so the web globals install before any other
  // per-workflow module (interceptors, then the user's workflows) evaluates; the
  // model auto-route follows it (it imports ADK) and precedes user code.
  t.deepEqual(workflowInterceptorModules, [
    require.resolve('../load-polyfills'),
    require.resolve('../auto-route'),
    'user-interceptors',
    require.resolve('../absorbed-failure'),
  ]);
  // Both side-effect modules satisfy the documented interceptor-module contract
  // (export an `interceptors` factory) while registering nothing.
  t.deepEqual(polyfillInterceptors(), {});
  t.deepEqual(autoRouteInterceptors(), {});
});

test('autoRouteModels: false leaves the auto-route module out of the bundle', (t) => {
  const plugin = new GoogleAdkPlugin({ autoRouteModels: false });
  const { workflowInterceptorModules } = plugin.configureBundler({
    workflowsPath: 'wf',
    workflowInterceptorModules: ['user-interceptors'],
  } as BundleOptions);
  t.deepEqual(workflowInterceptorModules, [
    require.resolve('../load-polyfills'),
    'user-interceptors',
    require.resolve('../absorbed-failure'),
  ]);
});

test('configureBundler appends the sandbox-compat plugin, preserving a user hook', (t) => {
  const userHook = (config: WebpackConfigLike): WebpackConfigLike => {
    config.plugins = [...(config.plugins ?? []), { name: 'user-plugin' }];
    return config;
  };
  const plugin = new GoogleAdkPlugin();
  const { webpackConfigHook } = plugin.configureBundler({
    workflowsPath: 'wf',
    webpackConfigHook: userHook,
  } as unknown as BundleOptions);
  t.is(typeof webpackConfigHook, 'function');

  const result = webpackConfigHook!({ plugins: [] } as never) as WebpackConfigLike;
  const names = (result.plugins ?? []).map((p) => p.name);
  t.deepEqual(names, ['user-plugin', 'google-adk-sandbox-compat']);
});

test('configureBundler aliases @opentelemetry/api to the copy @google/adk resolves', (t) => {
  const plugin = new GoogleAdkPlugin();
  const { webpackConfigHook } = plugin.configureBundler({ workflowsPath: 'wf' } as BundleOptions);
  const cfg = webpackConfigHook!({ plugins: [], resolve: { alias: { 'user-alias': '/user/alias' } } } as never) as {
    resolve?: { alias?: Record<string, unknown> };
  };
  const alias = cfg.resolve?.alias ?? {};

  // Every bare `@opentelemetry/api` request must land on ADK's own resolution,
  // so ADK's module-load tracer and the OTel interceptor's provider
  // registration share one api instance regardless of module evaluation order.
  // The exact-match (`$`) form leaves `@opentelemetry/api-logs` and subpath
  // imports untouched.
  const expected = createRequire(require.resolve('@google/adk')).resolve('@opentelemetry/api');
  t.is(alias['@opentelemetry/api$'], expected);
  t.false('@opentelemetry/api' in alias);
  t.is(alias['user-alias'], '/user/alias');
});

test('configureBundler pins the bare @google/adk specifier to the web build', (t) => {
  const plugin = new GoogleAdkPlugin();
  const { webpackConfigHook } = plugin.configureBundler({ workflowsPath: 'wf' } as BundleOptions);
  const cfg = webpackConfigHook!({ plugins: [], resolve: { alias: {} } } as never) as {
    resolve?: { alias?: Record<string, unknown> };
  };
  const alias = cfg.resolve?.alias ?? {};

  // Which `@google/adk` build the bundle gets must not depend on the consumer's
  // webpack target / browserslist: the sandbox always gets ADK's web surface,
  // and subpath imports (`@google/adk/tools/mcp`) are left alone by the `$` form.
  const expected = require.resolve('@google/adk/dist/web/index_web.js');
  t.is(alias['@google/adk$'], expected);
  t.false('@google/adk' in alias);
  t.true(expected.endsWith(path.join('dist', 'web', 'index_web.js')));
});

test('redirects crypto requests issued from inside @google/adk to the deterministic shim', (t) => {
  const resolve = sandboxCompatResolveHook();
  const shim = require.resolve('../crypto-shim');
  const adkIssuer = '/app/node_modules/@google/adk/dist/web/utils/env_aware_utils.js';
  // Both request forms, from an ADK file, land on the shim.
  t.is(resolve({ request: 'node:crypto', contextInfo: { issuer: adkIssuer } }), shim);
  t.is(resolve({ request: 'crypto', contextInfo: { issuer: adkIssuer } }), shim);
  // The issuer may only be known as a directory.
  t.is(resolve({ request: 'node:crypto', context: '/app/node_modules/@google/adk/dist/web/utils' }), shim);
  // Any other issuer keeps the bundler's policy: the scheme is stripped so the
  // bare-name `alias → false` applies, and nothing is redirected.
  t.is(
    resolve({ request: 'node:crypto', contextInfo: { issuer: '/app/node_modules/@google/genai/dist/web/x.js' } }),
    'crypto'
  );
  t.is(resolve({ request: 'crypto', contextInfo: { issuer: '/app/src/workflows.ts' } }), 'crypto');
  t.is(resolve({ request: 'node:crypto' }), 'crypto');
});

test("replaces ADK's own request for the broken apigee_llm.js web module", (t) => {
  const resolve = sandboxCompatResolveHook();
  const adkIssuer = '/app/node_modules/@google/adk/dist/web/models/registry.js';
  // ADK 2.0.0's `dist/web/models/apigee_llm.js` does not parse, so the request
  // is answered with the inert stand-in rather than the published file.
  t.true(resolve({ request: './apigee_llm.js', contextInfo: { issuer: adkIssuer } }).startsWith('data:'));
  // Issuer-gated like the crypto redirect: a same-named module of the user's
  // own resolves normally.
  t.is(
    resolve({ request: './apigee_llm.js', contextInfo: { issuer: '/app/src/models/registry.ts' } }),
    './apigee_llm.js'
  );
});

test('the api pin wins the bare specifier over user object-form alias entries', (t) => {
  const plugin = new GoogleAdkPlugin();
  const { webpackConfigHook } = plugin.configureBundler({ workflowsPath: 'wf' } as BundleOptions);
  const cfg = webpackConfigHook!({
    plugins: [],
    resolve: { alias: { '@opentelemetry/api': '/user/api-prefix', '@opentelemetry/api$': '/user/api-exact' } },
  } as never) as {
    resolve?: { alias?: Record<string, unknown> };
  };
  const alias = cfg.resolve?.alias ?? {};

  // Object-form aliases match in key insertion order, so the pin key must be
  // first and must keep the pin's target even when the user supplied the same
  // exact-match key. The user's prefix-form entry survives for subpath
  // imports but no longer captures the bare specifier — same semantics as the
  // array form below.
  const expected = createRequire(require.resolve('@google/adk')).resolve('@opentelemetry/api');
  t.is(Object.keys(alias)[0], '@opentelemetry/api$');
  t.is(alias['@opentelemetry/api$'], expected);
  t.is(alias['@opentelemetry/api'], '/user/api-prefix');
});

test('configureBundler prepends the api pin to an array-form user alias list', (t) => {
  const plugin = new GoogleAdkPlugin();
  const { webpackConfigHook } = plugin.configureBundler({ workflowsPath: 'wf' } as BundleOptions);
  const userEntry = { name: 'user-alias', alias: '/user/alias' };
  const cfg = webpackConfigHook!({ plugins: [], resolve: { alias: [userEntry] } } as never) as {
    resolve?: { alias?: Array<{ name: string; onlyModule?: boolean; alias: string }> };
  };
  const alias = cfg.resolve?.alias ?? [];

  // Alias entries resolve first-match-first in both forms, so the pin must
  // come first: it wins the bare specifier over any user entry, exactly as in
  // the object form.
  const expected = createRequire(require.resolve('@google/adk')).resolve('@opentelemetry/api');
  t.deepEqual(alias[0], { name: '@opentelemetry/api', onlyModule: true, alias: expected });
  t.deepEqual(alias[1], {
    name: '@google/adk',
    onlyModule: true,
    alias: require.resolve('@google/adk/dist/web/index_web.js'),
  });
  t.is(alias[2], userEntry);
});

test('configureWorker registers model activities, plus the four MCP activities per toolset', (t) => {
  const modelOnly = new GoogleAdkPlugin().configureWorker({ taskQueue: 'tq' } as WorkerOptions).activities as Record<
    string,
    unknown
  >;
  t.is(typeof modelOnly['adk-invokeModel'], 'function');
  t.is(typeof modelOnly['adk-invokeModelStreaming'], 'function');

  const withMcp = new GoogleAdkPlugin({ mcpToolsets: { weather: mockMCPToolset([]) } });
  const activities = withMcp.configureWorker({ taskQueue: 'tq' } as WorkerOptions).activities as Record<
    string,
    unknown
  >;
  t.is(typeof activities['weather-listTools'], 'function');
  t.is(typeof activities['weather-callTool'], 'function');
  t.is(typeof activities['weather-listResources'], 'function');
  t.is(typeof activities['weather-readResource'], 'function');
});

test('configureWorker refuses to replace a different Activity with the same name', async (t) => {
  const provider = fakeModelProvider();
  const weather = mockMCPToolset([]);
  const base = { taskQueue: 'tq' } as WorkerOptions;
  const duplicate = (name: string) => ({
    instanceOf: TypeError,
    message: new RegExp(`^Duplicate registration of Activity '${name}': GoogleAdkPlugin would replace`),
  });

  // A caller Activity, another instance's model provider, or another instance's
  // toolset factory under the same name is refused.
  const callerActivities = { ...base, activities: { 'adk-invokeModel': async () => 'caller' } };
  t.throws(() => new GoogleAdkPlugin().configureWorker(callerActivities), duplicate('adk-invokeModel'));
  const withProvider = new GoogleAdkPlugin({ modelProvider: provider }).configureWorker(base);
  t.throws(() => new GoogleAdkPlugin().configureWorker(withProvider), duplicate('adk-invokeModel'));
  const withToolset = new GoogleAdkPlugin({ mcpToolsets: { weather } }).configureWorker(base);
  const otherToolset = new GoogleAdkPlugin({ mcpToolsets: { weather: mockMCPToolset([]) } });
  t.throws(() => otherToolset.configureWorker(withToolset), duplicate('weather-listTools'));
  // Another operation built from the same provider, registered under this name.
  const streaming = (withProvider.activities as Record<string, unknown>)['adk-invokeModelStreaming'];
  const aliased = { ...base, activities: { 'adk-invokeModel': streaming } };
  t.throws(() => new GoogleAdkPlugin({ modelProvider: provider }).configureWorker(aliased), duplicate('adk-invokeModel'));
  // The provider is read once, at construction: changing the options object
  // afterwards does not change what the first plugin registered.
  const opts: GoogleAdkPluginOptions = { modelProvider: provider };
  const first = new GoogleAdkPlugin(opts);
  opts.modelProvider = fakeModelProvider();
  t.throws(() => new GoogleAdkPlugin(opts).configureWorker(first.configureWorker(base)), duplicate('adk-invokeModel'));
  // The provider is still called as a method of its options object, so the
  // object is part of what the model Activities depend on: one provider method
  // on two options objects is refused, as is one provider in two objects.
  function viaDelegate(this: { delegate: GoogleAdkPluginOptions['modelProvider'] }, model: string) {
    return this.delegate!(model);
  }
  const reply = (text: string) => fakeModelProvider([{ content: { role: 'model', parts: [{ text }] } }]);
  const methodOpts = { delegate: reply('from delegate'), modelProvider: viaDelegate };
  const otherMethodOpts = { delegate: reply('from another delegate'), modelProvider: viaDelegate };
  t.throws(
    () => new GoogleAdkPlugin(otherMethodOpts).configureWorker(new GoogleAdkPlugin(methodOpts).configureWorker(base)),
    duplicate('adk-invokeModel')
  );
  t.throws(() => new GoogleAdkPlugin({ modelProvider: provider }).configureWorker(withProvider), duplicate('adk-invokeModel'));
  const invokeArgs = { model: 'm', request: { model: 'm', contents: [{ role: 'user', parts: [{ text: 'hi' }] }], config: {} } } as InvokeModelArgs;
  const methodActivities = new GoogleAdkPlugin(methodOpts).configureWorker(base).activities as ModelActivities;
  const responses = (await new MockActivityEnvironment().run(methodActivities['adk-invokeModel'], invokeArgs)) as LlmResponse[];
  t.is(responses[0]?.content?.parts?.[0]?.text, 'from delegate');

  // configureWorker accepts the same instance twice (NativeConnection and
  // Worker) and the same options object twice.
  const plugin = new GoogleAdkPlugin({ modelProvider: provider, mcpToolsets: { weather } });
  t.notThrows(() => plugin.configureWorker(plugin.configureWorker(base)));
  t.notThrows(() => new GoogleAdkPlugin().configureWorker(new GoogleAdkPlugin().configureWorker(base)));
  t.notThrows(() => new GoogleAdkPlugin({ mcpToolsets: { weather } }).configureWorker(withToolset));
  const shared: GoogleAdkPluginOptions = { modelProvider: provider, mcpToolsets: { weather } };
  t.notThrows(() => new GoogleAdkPlugin(shared).configureWorker(new GoogleAdkPlugin(shared).configureWorker(base)));
  t.notThrows(() => new GoogleAdkPlugin(methodOpts).configureWorker(new GoogleAdkPlugin(methodOpts).configureWorker(base)));
  // Different toolset names from two instances are both kept.
  const both = new GoogleAdkPlugin({ mcpToolsets: { news: mockMCPToolset([]) } })
    .configureWorker(withToolset)
    .activities as Record<string, unknown>;
  t.is(typeof both['weather-callTool'], 'function');
  t.is(typeof both['news-callTool'], 'function');
});
