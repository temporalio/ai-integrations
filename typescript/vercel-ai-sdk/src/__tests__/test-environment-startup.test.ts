import { mock } from 'node:test';
import test from 'ava';
import { startLocalWithRetry } from './helpers/environment';

const STARTUP_TIMEOUT =
  'Failed to start ephemeral server: ephemeral server at 127.0.0.1:49225 did not start within 5s. ' +
  "Make sure another download isn't stuck and delete the temp file.: Server connection error: Connection refused";

test.afterEach.always(() => mock.reset());

for (const timeouts of [0, 1, 2]) {
  test.serial(`local server starts after ${timeouts} startup timeouts`, async (t) => {
    const environment = {};
    // Configuration stays opaque to the retry policy, just as in the Python helper.
    const options = {
      namespace: 'startup-test',
      extraArgs: ['--dynamic-config-value', 'test.option=50'],
    };
    let attempts = 0;
    const start = mock.fn(async (_options: typeof options) => {
      if (attempts++ < timeouts) throw new Error(STARTUP_TIMEOUT);
      return environment;
    });
    mock.method(console, 'warn', () => undefined);

    t.is(await startLocalWithRetry(() => start(options)), environment);
    t.is(start.mock.callCount(), timeouts + 1);
    for (const call of start.mock.calls) {
      t.is(call.arguments[0], options);
    }
  });
}

test.serial('local server startup stops after three timeouts', async (t) => {
  const error = new Error(STARTUP_TIMEOUT);
  const start = mock.fn(async () => {
    throw error;
  });
  mock.method(console, 'warn', () => undefined);

  t.is(await t.throwsAsync(startLocalWithRetry(start)), error);
  t.is(start.mock.callCount(), 3);
});

for (const [name, error] of [
  ['invalid configuration', new Error('Failed to start ephemeral server: invalid configuration')],
  ['download failure', new Error('Failed to start ephemeral server: download failed')],
  ['connection failure', new Error('Failed to start ephemeral server: Connection refused')],
  ['unrelated timeout', new Error('request did not start within 5s.')],
  ['cancellation', new DOMException('Cancelled', 'AbortError')],
] as const) {
  test.serial(`local server startup does not retry ${name}`, async (t) => {
    const start = mock.fn(async () => {
      throw error;
    });

    t.is(await t.throwsAsync(startLocalWithRetry(start)), error);
    t.is(start.mock.callCount(), 1);
  });
}
