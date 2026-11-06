import test from 'ava';
import { LangGraphPlugin } from '../index';
import { registry } from './graphs';
import { execute, setupTestEnv, uid, withWorker } from './helpers';
import { interruptWorkflow } from './workflows';

const env = setupTestEnv(test);

// LangGraph.js has no `version: 'v2'` invoke option; `getState()` is how it
// reports interrupts apart from the state values.
for (const inspect of ['result', 'state'] as const) {
  test(`an Activity node interrupts and resumes, checked through the ${inspect}`, async (t) => {
    const taskQueue = uid('interrupt');
    const result = await withWorker(env(), { taskQueue, plugin: new LangGraphPlugin(registry.interruptGraph!) }, () =>
      execute(env(), interruptWorkflow, { taskQueue, args: ['', inspect] })
    );
    t.deepEqual(result, { value: 'yes' });
  });
}
