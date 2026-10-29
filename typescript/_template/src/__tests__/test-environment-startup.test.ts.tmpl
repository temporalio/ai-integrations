import { mock } from 'node:test';
import test from 'ava';
import { startLocalWithRetry } from './helpers/environment';

const STARTUP_PREFIX = 'Failed to start ephemeral server: ';
const STARTUP_TIMEOUT =
  STARTUP_PREFIX +
  'ephemeral server at 127.0.0.1:49225 did not start within 5s. ' +
  "Make sure another download isn't stuck and delete the temp file.: Server connection error: Connection refused";
const DOWNLOAD_FAILURE =
  STARTUP_PREFIX + 'failed to download ephemeral server executable: error sending request';
const RETRYABLE_ERRORS = [
  ['startup timeout', STARTUP_TIMEOUT],
  ['download failure', DOWNLOAD_FAILURE],
  ...[
    'could not read tar archive',
    'could not read tar archive entry',
    'could not extract tar archive entry',
    'could not read ZIP archive',
    'could not extract ZIP archive entry',
  ].map((message) => [message, STARTUP_PREFIX + 'invalid ephemeral server download: ' + message] as const),
] as const;

test.afterEach.always(() => mock.reset());

function mockRetryDelay() {
  const originalSetTimeout = globalThis.setTimeout;
  return mock.method(globalThis, 'setTimeout', (callback: () => void, _delay?: number) => originalSetTimeout(callback, 0));
}

for (const [name, message] of RETRYABLE_ERRORS) {
  for (const failures of [0, 1, 2]) {
    test.serial(`local server starts after ${failures} ${name} failures`, async (t) => {
      const environment = {};
      // Configuration stays opaque to the retry policy, just as in the Python helper.
      const options = {
        namespace: 'startup-test',
        extraArgs: ['--dynamic-config-value', 'test.option=50'],
      };
      let attempts = 0;
      const start = mock.fn(async (_options: typeof options) => {
        if (attempts++ < failures) throw new Error(message);
        return environment;
      });
      const warn = mock.method(console, 'warn', () => undefined);
      const delay = mockRetryDelay();

      t.is(await startLocalWithRetry(() => start(options)), environment);
      t.is(start.mock.callCount(), failures + 1);
      t.is(warn.mock.callCount(), failures);
      t.deepEqual(
        delay.mock.calls.map((call) => call.arguments[1]),
        Array.from({ length: failures }, () => 1000)
      );
      for (const call of start.mock.calls) {
        t.is(call.arguments[0], options);
      }
    });
  }

  test.serial(`local server stops after three ${name} failures`, async (t) => {
    const error = new Error(message);
    const start = mock.fn(async () => {
      throw error;
    });
    const warn = mock.method(console, 'warn', () => undefined);
    const delay = mockRetryDelay();

    t.is(await t.throwsAsync(startLocalWithRetry(start)), error);
    t.is(start.mock.callCount(), 3);
    t.is(warn.mock.callCount(), 2);
    t.deepEqual(delay.mock.calls.map((call) => call.arguments[1]), [1000, 1000]);
  });
}

test.serial('different startup errors share the three-attempt limit', async (t) => {
  const errors = RETRYABLE_ERRORS.slice(0, 3).map(([, message]) => new Error(message));
  let attempt = 0;
  const start = mock.fn(async () => {
    throw errors[attempt++];
  });
  mock.method(console, 'warn', () => undefined);
  const delay = mockRetryDelay();

  t.is(await t.throwsAsync(startLocalWithRetry(start)), errors[2]);
  t.is(start.mock.callCount(), 3);
  t.deepEqual(delay.mock.calls.map((call) => call.arguments[1]), [1000, 1000]);
});

for (const [name, error] of [
  ['missing executable', new Error(STARTUP_PREFIX + 'ephemeral server executable does not exist: /bad')],
  ['process launch failure', new Error(STARTUP_PREFIX + 'failed to start ephemeral server: permission denied')],
  ['unsupported architecture', new Error(STARTUP_PREFIX + 'invalid ephemeral server download: unsupported architecture: other')],
  ['unsupported archive format', new Error(STARTUP_PREFIX + 'invalid ephemeral server download: archive URL has unsupported format')],
  ['invalid configuration', new Error(STARTUP_PREFIX + 'invalid configuration')],
  ['native connection failure', new Error('Failed to connect to Temporal server: connection refused')],
  ['client connection failure', new Error('Failed to connect before the deadline')],
  ['unrelated timeout', new Error('request did not start within 5s.')],
  ['programming error', new TypeError('invalid startup configuration')],
  ['DOM cancellation', new DOMException('Cancelled', 'AbortError')],
  ['Error cancellation', Object.assign(new Error(STARTUP_TIMEOUT), { name: 'AbortError' })],
  ['non-Error rejection', { message: DOWNLOAD_FAILURE }],
] as const) {
  test.serial(`local server startup does not retry ${name}`, async (t) => {
    const start = mock.fn(async () => {
      throw error;
    });
    const warn = mock.method(console, 'warn', () => undefined);
    const delay = mockRetryDelay();

    t.is(await startLocalWithRetry(start).catch((caught: unknown) => caught), error);
    t.is(start.mock.callCount(), 1);
    t.is(warn.mock.callCount(), 0);
    t.is(delay.mock.callCount(), 0);
  });
}

test.serial('unexpected errors after a retry propagate immediately', async (t) => {
  const error = new TypeError('invalid startup configuration');
  const errors = [new Error(DOWNLOAD_FAILURE), error];
  let attempt = 0;
  const start = mock.fn(async () => {
    throw errors[attempt++];
  });
  mock.method(console, 'warn', () => undefined);
  const delay = mockRetryDelay();

  t.is(await t.throwsAsync(startLocalWithRetry(start)), error);
  t.is(start.mock.callCount(), 2);
  t.deepEqual(delay.mock.calls.map((call) => call.arguments[1]), [1000]);
});

test.serial('cancellation during the retry delay propagates immediately', async (t) => {
  const error = new DOMException('Cancelled', 'AbortError');
  const start = mock.fn(async () => {
    throw new Error(DOWNLOAD_FAILURE);
  });
  mock.method(console, 'warn', () => undefined);
  mock.method(globalThis, 'setTimeout', () => {
    throw error;
  });

  t.is(await t.throwsAsync(startLocalWithRetry(start)), error);
  t.is(start.mock.callCount(), 1);
});
