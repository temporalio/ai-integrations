/**
 * Client marker runs match the Python plugin: they span the call, record no Temporal args, end with
 * `{ status: 'ok' }` or the call's error, and propagate the context Python propagates for each call.
 *
 * @module
 */

import test from 'ava';
import { traceable, getCurrentRunTree } from 'langsmith/traceable';
import { CancelledFailure, type Payload } from '@temporalio/common';
import type { WorkflowClientInterceptor } from '@temporalio/client';

import { createClientInterceptor } from '../client-interceptor';
import { readContextHeader } from '../propagation';
import { InMemoryRunCollector, idPath } from './helpers';

process.env.LANGSMITH_TRACING = 'true';
// Keep langsmith callbacks synchronous so emission completes within the test.
process.env.LANGCHAIN_CALLBACKS_BACKGROUND = 'false';

function interceptor(collector: InMemoryRunCollector): WorkflowClientInterceptor {
  return createClientInterceptor({ client: collector.asClient(), addTemporalRuns: true });
}

/** The run id a propagated header points at as the remote parent. */
function propagatedParentId(headers: Record<string, Payload> | undefined): string | undefined {
  const context = readContextHeader(headers);
  return context && idPath(context['langsmith-trace']).at(-1);
}

/** Call `fn` inside an app `traceable` run, returning that run's id. */
async function underAppRun(collector: InMemoryRunCollector, fn: () => Promise<unknown>): Promise<string> {
  return traceable(
    async () => {
      await fn();
      return getCurrentRunTree().id;
    },
    { name: 'app', client: collector.asClient() }
  )();
}

const CALLS: Record<string, (i: WorkflowClientInterceptor, next: (input: unknown) => Promise<unknown>) => Promise<unknown>> = {
  'StartWorkflow:W': (i, next) =>
    i.startWithDetails!({ headers: {}, workflowType: 'W', options: { args: ['a'] } } as never, next as never),
  'SignalWithStartWorkflow:W': (i, next) =>
    i.signalWithStart!({ headers: {}, workflowType: 'W', signalName: 's', signalArgs: ['a'], options: { args: [] } } as never, next as never),
  'SignalWorkflow:s': (i, next) => i.signal!({ headers: {}, signalName: 's', args: ['a'] } as never, next as never),
  'QueryWorkflow:q': (i, next) => i.query!({ headers: {}, queryType: 'q', args: ['a'] } as never, next as never),
  'StartWorkflowUpdate:u': (i, next) => i.startUpdate!({ headers: {}, updateName: 'u', args: ['a'] } as never, next as never),
  'StartUpdateWithStartWorkflow:W': (i, next) =>
    i.startUpdateWithStart!(
      { workflowType: 'W', workflowStartHeaders: {}, updateName: 'u', updateArgs: ['a'], updateHeaders: {} } as never,
      next as never
    ),
};

for (const [marker, call] of Object.entries(CALLS)) {
  test.serial(`${marker}: marker spans the call, records no Temporal args, and ends ok`, async (t) => {
    const collector = new InMemoryRunCollector();
    let duringCall: { posted: boolean; ended: boolean } | undefined;
    await call(interceptor(collector), async () => {
      const run = collector.byName(marker);
      duringCall = { posted: run !== undefined, ended: run?.end_time !== undefined };
      return 'result';
    });
    t.deepEqual(duringCall, { posted: true, ended: false });
    const run = collector.byName(marker)!;
    t.deepEqual(run.inputs, {});
    t.deepEqual(run.outputs, { status: 'ok' });
    t.truthy(run.end_time);
    t.falsy(run.error);
  });

  test.serial(`${marker}: marker records the call's error and rethrows it`, async (t) => {
    const collector = new InMemoryRunCollector();
    await t.throwsAsync(
      call(interceptor(collector), async () => {
        throw new TypeError('boom');
      }),
      { message: 'boom' }
    );
    const run = collector.byName(marker)!;
    t.is(run.error, 'TypeError: boom');
    t.truthy(run.end_time);
  });

  test.serial(`${marker}: marker ends cancelled, not errored, when the call is cancelled`, async (t) => {
    const collector = new InMemoryRunCollector();
    await t.throwsAsync(
      call(interceptor(collector), async () => {
        throw new CancelledFailure('cancelled');
      })
    );
    const run = collector.byName(marker)!;
    t.deepEqual(run.outputs, { status: 'cancelled' });
    t.falsy(run.error);
  });
}

test.serial('signalWithStart propagates the caller run, so the remote handler is a sibling of the marker', async (t) => {
  const collector = new InMemoryRunCollector();
  let sent: Record<string, Payload> | undefined;
  const appId = await underAppRun(collector, () =>
    CALLS['SignalWithStartWorkflow:W']!(interceptor(collector), async (input) => {
      sent = (input as { headers: Record<string, Payload> }).headers;
    })
  );
  t.is(propagatedParentId(sent), appId);
});

test.serial('signal propagates its marker, so the remote handler nests under it', async (t) => {
  const collector = new InMemoryRunCollector();
  let sent: Record<string, Payload> | undefined;
  await underAppRun(collector, () =>
    CALLS['SignalWorkflow:s']!(interceptor(collector), async (input) => {
      sent = (input as { headers: Record<string, Payload> }).headers;
    })
  );
  t.is(propagatedParentId(sent), collector.byName('SignalWorkflow:s')!.id);
});

test.serial('startUpdateWithStart propagates the caller run into both start and update headers', async (t) => {
  const collector = new InMemoryRunCollector();
  let sent: { workflowStartHeaders: Record<string, Payload>; updateHeaders: Record<string, Payload> } | undefined;
  const appId = await underAppRun(collector, () =>
    CALLS['StartUpdateWithStartWorkflow:W']!(interceptor(collector), async (input) => {
      sent = input as typeof sent;
    })
  );
  t.is(propagatedParentId(sent?.workflowStartHeaders), appId);
  t.is(propagatedParentId(sent?.updateHeaders), appId);
});
