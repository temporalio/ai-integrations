/**
 * The deterministic `crypto` shim: LangChain's `uuid` and the sandbox `crypto`
 * global draw from a named Workflow random stream, so IDs are well-formed and
 * regenerate identically on replay.
 */

import test from 'ava';
import { LangGraphPlugin } from '../index';
import { registry } from './graphs';
import { execute, setupTestEnv, uid, withWorker } from './helpers';
import { cryptoShimProbe } from './workflows';

const env = setupTestEnv(test);

test('IDs drawn in the Workflow are UUIDs that regenerate identically on replay', async (t) => {
  const taskQueue = uid('crypto-shim');
  const plugin = new LangGraphPlugin(registry['twoNodes.activity']!);
  // The Worker caches no Workflows, so the final task replays the whole
  // history: the IDs it returns are regenerated, while `echoed` is the
  // Activity result recorded on the first task.
  const ids = await withWorker(env(), { taskQueue, plugin }, () =>
    execute(env(), cryptoShimProbe, { taskQueue, args: [] })
  );
  t.regex(ids.v4, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  t.regex(ids.v7, /^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  t.regex(ids.direct, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  t.regex(ids.bytes, /^[0-9a-f]{16}$/);
  t.not(ids.v4, ids.direct);
  const { echoed, ...drawn } = ids;
  t.is(echoed, `${JSON.stringify(drawn)}ab`);
});
