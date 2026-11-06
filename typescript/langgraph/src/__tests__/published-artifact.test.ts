/**
 * Regression test that consumes the COMPILED `lib` the way a published-package
 * user does: resolved BY PACKAGE NAME (`@temporalio/langgraph` and its
 * `/workflow` subpath) via Node package self-reference, never via relative
 * `../` source paths.
 */

import { createRequire } from 'node:module';
import test from 'ava';

test('cjsRequireExposesPublicExports', (t) => {
  const require = createRequire(__filename);

  const main = require('@temporalio/langgraph');
  t.is(typeof main.LangGraphPlugin, 'function');
  t.is(typeof main.defineLangGraph, 'function');
  t.is(typeof main.LangGraphDefinitions, 'function');
  t.false('UNKNOWN_VALUE_TYPE_FAILURE_TYPE' in main);

  const workflow = require('@temporalio/langgraph/workflow');
  t.is(typeof workflow.defineLangGraph, 'function');
  t.is(typeof workflow.LangGraphDefinitions, 'function');
  t.is(typeof workflow.cache, 'function');
  t.is(workflow.UNKNOWN_VALUE_TYPE_FAILURE_TYPE, 'LangGraphUnknownValueType');
  t.false('LangGraphPlugin' in workflow);
  t.is(workflow.defineLangGraph, main.defineLangGraph);
});
