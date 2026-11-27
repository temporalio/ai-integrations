import test from 'ava';

import { createTestEnvironment } from './helpers/environment';

test('live Temporal tests require a shared server address', async (t) => {
  const address = process.env.TEMPORAL_TEST_ADDRESS;
  delete process.env.TEMPORAL_TEST_ADDRESS;
  t.teardown(() => {
    if (address !== undefined) {
      process.env.TEMPORAL_TEST_ADDRESS = address;
    }
  });
  await t.throwsAsync(createTestEnvironment(), { message: /make test.*TEMPORAL_TEST_ADDRESS/ });
});
