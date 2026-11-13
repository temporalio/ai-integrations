import test from 'ava';
import { DefaultModelRetryStrategy } from '@strands-agents/sdk';
import { TemporalAgent } from '..';

test('TemporalAgent accepts an omitted retryStrategy', (t) => {
  t.notThrows(() => new TemporalAgent());
});

test('TemporalAgent accepts retryStrategy: undefined', (t) => {
  t.notThrows(() => new TemporalAgent({ retryStrategy: undefined }));
});

test('TemporalAgent accepts retryStrategy: null as its error message recommends', (t) => {
  t.notThrows(() => new TemporalAgent({ retryStrategy: null }));
});

test('TemporalAgent rejects a Strands retry strategy at construction', (t) => {
  t.throws(
    () => {
      // @ts-expect-error Temporal owns retries; only null or undefined is supported.
      new TemporalAgent({ retryStrategy: new DefaultModelRetryStrategy() });
    },
    { message: /TemporalAgent disables Strands retries; configure retries via activityOptions\.retry/ }
  );
});
