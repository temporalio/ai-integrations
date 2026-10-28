import { mock } from 'node:test';
import test from 'ava';
import { startLocalWithRetry } from './helpers/environment';

test.afterEach.always(() => mock.reset());

for (const failures of [0, 1, 2]) {
  test.serial(`local server starts after ${failures} setup failures`, async (t) => {
    const environment = {};
    // Configuration stays opaque to the retry policy, just as in the Python helper.
    const options = {
      namespace: 'startup-test',
      extraArgs: ['--dynamic-config-value', 'test.option=50'],
    };
    let attempts = 0;
    const start = mock.fn(async (_options: typeof options) => {
      if (attempts++ < failures) throw new TypeError('server setup failed');
      return environment;
    });
    const warn = mock.method(console, 'warn', () => undefined);

    t.is(await startLocalWithRetry(() => start(options)), environment);
    t.is(start.mock.callCount(), failures + 1);
    t.is(warn.mock.callCount(), failures);
    for (const call of start.mock.calls) {
      t.is(call.arguments[0], options);
    }
  });
}

test.serial('different startup errors share the three-attempt limit', async (t) => {
  const error = new Error('server startup failed');
  const errors = [new Error('download failed'), new TypeError('invalid archive'), error];
  let attempt = 0;
  const start = mock.fn(async () => {
    throw errors[attempt++];
  });
  const warn = mock.method(console, 'warn', () => undefined);

  t.is(await t.throwsAsync(startLocalWithRetry(start)), error);
  t.is(start.mock.callCount(), 3);
  t.is(warn.mock.callCount(), 2);
});

test.serial('local server startup retries non-Error rejections', async (t) => {
  const error = { message: 'server setup failed' };
  const start = mock.fn(async () => {
    throw error;
  });
  const warn = mock.method(console, 'warn', () => undefined);

  t.is(await startLocalWithRetry(start).catch((caught: unknown) => caught), error);
  t.is(start.mock.callCount(), 3);
  t.is(warn.mock.callCount(), 2);
});

for (const error of [
  new DOMException('Cancelled', 'AbortError'),
  Object.assign(new Error('Cancelled'), { name: 'AbortError' }),
]) {
  test.serial(`local server startup does not retry cancellation from ${error.constructor.name}`, async (t) => {
    const start = mock.fn(async () => {
      throw error;
    });
    const warn = mock.method(console, 'warn', () => undefined);

    t.is(await t.throwsAsync(startLocalWithRetry(start)), error);
    t.is(start.mock.callCount(), 1);
    t.is(warn.mock.callCount(), 0);
  });
}
