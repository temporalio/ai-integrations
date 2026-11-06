import { readFileSync } from 'node:fs';
import path from 'node:path';
import test from 'ava';
import { START, StateGraph, entrypoint, task } from '@langchain/langgraph';
import { internalsOf } from '../definitions';
import { LangGraphPlugin, defineLangGraph } from '../index';
import { patchConfig } from '../runtime';
import { ValueState, lambdaNodeGraph } from './graphs';

const done = async () => ({ value: 'done' });

function graphWith(metadata?: Record<string, unknown>, options: Record<string, unknown> = {}) {
  return new StateGraph(ValueState).addNode('node', done, { metadata, ...options }).addEdge(START, 'node');
}

test('a node that is not a RunnableCallable is rejected', (t) => {
  t.throws(() => defineLangGraph({ graphs: { validation: lambdaNodeGraph } }), {
    message: /must be a function \(a RunnableCallable\)/,
  });
});

test('an invalid executeIn is rejected', (t) => {
  t.throws(() => defineLangGraph({ graphs: { validation: graphWith({ executeIn: 'bogus' }) } }), {
    message: /Invalid executeIn value/,
  });
});

test('a graph node without executeIn is rejected', (t) => {
  t.throws(() => defineLangGraph({ graphs: { validation: graphWith() } }), {
    message: /missing required 'executeIn'/,
  });
});

test('a Functional API task without executeIn is rejected', (t) => {
  const myTask = task('my_task', (x: number) => x + 1);
  t.throws(() => defineLangGraph({ tasks: [myTask] }), { message: /missing required 'executeIn'/ });
});

test('executeIn in defaultActivityOptions is rejected', (t) => {
  t.throws(() => defineLangGraph({ defaultActivityOptions: { executeIn: 'activity' } as never }), {
    message: /cannot be set in defaultActivityOptions/,
  });
});

test('a node retryPolicy is rejected', (t) => {
  const graph = graphWith({ executeIn: 'activity' }, { retryPolicy: { maxAttempts: 3 } });
  t.throws(() => defineLangGraph({ graphs: { validation: graph } }), { message: /retryPolicy/ });
});

test('a task retry policy is rejected', (t) => {
  const myTask = task({ name: 'my_task', retry: { maxAttempts: 3 } }, (x: number) => x + 1);
  t.throws(
    () => defineLangGraph({ tasks: [myTask], activityOptions: { my_task: { executeIn: 'activity' } } }),
    { message: /retry policy/ }
  );
});

test('a task registered twice is rejected', (t) => {
  const first = task('my_task', (x: number) => x + 1);
  const second = task('my_task', (x: number) => x + 2);
  t.throws(
    () => defineLangGraph({ tasks: [first, second], activityOptions: { my_task: { executeIn: 'activity' } } }),
    { message: /registered more than once/ }
  );
});

test('an entrypoint registered under another name is rejected', (t) => {
  const program = entrypoint('real_name', async (x: number) => x);
  t.throws(() => defineLangGraph({ entrypoints: { other: program } }), { message: /under its own name/ });
});

test('both summary and summaryFn in defaultActivityOptions are rejected', (t) => {
  t.throws(() => defineLangGraph({ defaultActivityOptions: { summary: 's', summaryFn: () => 'f' } }), {
    message: /defaultActivityOptions/,
  });
});

test('both summary and summaryFn on a node are rejected', (t) => {
  const graph = graphWith({ executeIn: 'activity', summary: 'static', summaryFn: () => 'dynamic' });
  t.throws(() => defineLangGraph({ graphs: { 'summary-graph': graph } }), { message: /not both/ });
});

test('graph() outside a Workflow is rejected', (t) => {
  const definitions = defineLangGraph({ graphs: { validation: graphWith({ executeIn: 'activity' }) } });
  t.throws(() => definitions.graph('validation'), { message: /must be called from inside a Workflow/ });
});

test('the plugin reports the package version', (t) => {
  const { version } = JSON.parse(readFileSync(path.resolve(__dirname, '../../package.json'), 'utf8'));
  t.is(new LangGraphPlugin(defineLangGraph()).version, version);
});

test("a run without LangGraph's task scheduler fails instead of running tasks inline", (t) => {
  const myTask = task('my_task', (x: number) => x + 1);
  const internals = internalsOf(
    defineLangGraph({ tasks: [myTask], activityOptions: { my_task: { executeIn: 'activity' } } })
  );
  t.throws(() => patchConfig(internals, { configurable: {} }, undefined), { message: /task scheduler/ });

  const scheduler = () => undefined;
  const patched = patchConfig(internals, { configurable: { __pregel_call: scheduler } }, undefined);
  t.not(patched.configurable.__pregel_call, scheduler);
  t.is(typeof patched.configurable.__pregel_call, 'function');
});
