import test from 'ava';
import { restoreConfig } from '../activities';
import { serializeConfig } from '../runtime';

test('the serializable subset of a RunnableConfig crosses the Activity boundary', (t) => {
  const send = () => undefined;
  const payload = serializeConfig({
    tags: ['t'],
    metadata: { user: 'alice' },
    runName: 'my-run',
    runId: '1ef6c0a0-0000-4000-8000-000000000000',
    recursionLimit: 5,
    context: { tenant: 'acme' },
    store: {},
    callbacks: [],
    configurable: {
      thread_id: '1',
      checkpoint_ns: 'node:1',
      __pregel_previous: { value: 'before' },
      __pregel_send: send,
      __pregel_scratchpad: { resume: ['yes'], nullResume: undefined },
      __pregel_checkpointer: {},
    },
  });
  t.deepEqual(JSON.parse(JSON.stringify(payload)), payload);

  const config = restoreConfig(payload, undefined);
  t.deepEqual(config.tags, ['t']);
  t.deepEqual(config.metadata, { user: 'alice' });
  t.is(config.runName, 'my-run');
  t.is(config.runId, '1ef6c0a0-0000-4000-8000-000000000000');
  t.is(config.recursionLimit, 5);
  t.deepEqual(config.context, { tenant: 'acme' });
  t.false('store' in config);
  t.false('callbacks' in config);
  const configurable = config.configurable as Record<string, unknown>;
  t.is(configurable.thread_id, '1');
  t.is(configurable.checkpoint_ns, 'node:1');
  t.deepEqual(configurable.__pregel_previous, { value: 'before' });
  t.true(configurable.__pregel_checkpointer);
  t.not(configurable.__pregel_send, send);
  t.deepEqual((configurable.__pregel_scratchpad as { resume: unknown[] }).resume, ['yes']);
});
