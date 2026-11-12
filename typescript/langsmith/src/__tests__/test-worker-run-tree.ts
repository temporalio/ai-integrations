import test from 'ava';
import { address } from 'langsmith';
import { RunTree } from 'langsmith/run_trees';
import { getCurrentRunTree, traceable, withRunTree } from 'langsmith/traceable';
import * as nexus from 'nexus-rpc';
import { createActivityInboundInterceptor, createNexusInboundInterceptor } from '../activity-interceptor';
import type { EmitterConfig } from '../sinks';
import { InMemoryRunCollector } from './helpers';

process.env.LANGSMITH_TRACING = 'true';
process.env.LANGCHAIN_CALLBACKS_BACKGROUND = 'false';

type Inbound = 'activity' | 'nexus-start' | 'nexus-cancel';
const INBOUNDS: Inbound[] = ['activity', 'nexus-start', 'nexus-cancel'];
const ADDRESS = address.agent('support', 'production');

async function invoke(inbound: Inbound, config: EmitterConfig, body: () => Promise<void>): Promise<void> {
  if (inbound === 'activity') {
    await createActivityInboundInterceptor(config)({} as never).execute!({ args: [], headers: {} }, body);
  } else {
    const interceptor = createNexusInboundInterceptor(config);
    const input = { ctx: { service: 'service', operation: 'operation', headers: {} } } as never;
    if (inbound === 'nexus-start') {
      await interceptor.startOperation!(input, async () => ({ result: nexus.HandlerStartOperationResult.sync(await body()) }));
    } else {
      await interceptor.cancelOperation!(input, body);
    }
  }
}

for (const inbound of INBOUNDS) {
  for (const mode of ['project', 'address'] as const) {
    test(`${inbound}: parentless ${mode} defaults create native roots without an ambient placeholder`, async (t) => {
      const collector = new InMemoryRunCollector();
      const config: EmitterConfig = {
        client: collector.asClient(), addTemporalRuns: false,
        ...(mode === 'project' ? { projectName: 'plugin-project' } : { address: ADDRESS }),
        replicas: [],
      };
      await invoke(inbound, config, async () => {
        t.is(getCurrentRunTree(true), undefined);
        await Promise.resolve();
        t.is(getCurrentRunTree(true), undefined);
        const native = new RunTree({ name: 'native-root' });
        // Stop before posting if a regression selected a real shared client.
        if (!t.is(native.client, config.client)) return;
        t.is(native.parent_run, undefined);
        t.deepEqual(native.replicas, []);
        await native.postRun();
        await native.end({});
        await native.patchRun();
        await traceable(async () => {
          const root = getCurrentRunTree();
          t.is(root.parent_run, undefined);
          t.is(root.client, config.client);
          await traceable(async () => {
            t.is(getCurrentRunTree().parent_run, root);
          }, { name: 'traceable-child' })();
        }, { name: 'traceable-root' })();
        t.is(getCurrentRunTree(true), undefined);
      });
      t.is(collector.records.length, 3);
      for (const run of collector.records) {
        t.is(run.project_name, mode === 'project' ? 'plugin-project' : undefined);
        t.is(run.address, mode === 'address' ? ADDRESS : undefined);
        t.truthy(run.end_time);
      }
      for (const name of ['native-root', 'traceable-root']) {
        const root = collector.byName(name)!;
        t.falsy(root.parent_run_id);
        t.is(root.trace_id, root.id);
        t.false(root.dotted_order!.includes('.'));
      }
      t.not(collector.byName('native-root')!.trace_id, collector.byName('traceable-root')!.trace_id);
    });
  }
}

test('worker defaults preserve explicit destinations, clients, replicas and native parent inheritance', async (t) => {
  const collector = new InMemoryRunCollector();
  const explicit = new InMemoryRunCollector();
  await invoke('activity', {
    client: collector.asClient(), addTemporalRuns: false, address: ADDRESS,
    replicas: [{ projectName: 'plugin-replica' }],
  }, async () => {
    const defaults = new RunTree({ name: 'defaults' });
    t.is(defaults.address, ADDRESS);
    t.deepEqual(defaults.replicas, [{ projectName: 'plugin-replica' }]);
    const native = new RunTree({ name: 'explicit', project_name: 'explicit-project', client: explicit.asClient(), replicas: [] });
    t.is(native.project_name, 'explicit-project');
    t.is(native.address, undefined);
    t.is(native.client, explicit.asClient());
    t.deepEqual(native.replicas, []);
    await withRunTree(native, () => traceable(async () => {
      const child = getCurrentRunTree();
      t.is(child.parent_run, native);
      t.is(child.project_name, 'explicit-project');
      t.is(child.address, undefined);
      t.is(child.client, native.client);
      t.deepEqual(child.replicas, []);
    }, { name: 'child' })());
    await traceable(async () => {}, {
      name: 'explicit-traceable', project_name: 'traceable-project', client: explicit.asClient(), replicas: [],
    })();
    t.throws(() => new RunTree({ name: 'conflict', project_name: 'project', address: ADDRESS }));
    t.throws(() => new RunTree({ name: 'invalid', address: '' as typeof ADDRESS }));
    await traceable(async () => {}, { name: 'disabled', tracingEnabled: false })();
  });
  t.deepEqual(collector.records, []);
  t.is(explicit.byName('explicit-traceable')?.project_name, 'traceable-project');
});

test('concurrent worker defaults stay local and restore the caller context after failure', async (t) => {
  const outside = new RunTree({ name: 'outside' });
  const collectors = [new InMemoryRunCollector(), new InMemoryRunCollector()];
  let ready = 0;
  let release!: () => void;
  const barrier = new Promise<void>((resolve) => { release = resolve; });
  await withRunTree(outside, async () => {
    await Promise.all(collectors.map((collector, index) => invoke(INBOUNDS[index]!, {
      client: collector.asClient(), addTemporalRuns: false, projectName: `project-${index}`, replicas: [],
    }, async () => {
      if (++ready === collectors.length) release();
      await barrier;
      t.is(getCurrentRunTree(true), undefined);
      if (!t.is(new RunTree({ name: 'probe' }).client, collector.asClient())) return;
      await traceable(async () => {}, { name: `root-${index}` })();
    })));
    t.is(getCurrentRunTree(), outside);
    await t.throwsAsync(invoke('nexus-cancel', {
      client: collectors[0]!.asClient(), addTemporalRuns: false, projectName: 'failed',
    }, async () => { throw new Error('handler failed'); }), { message: 'handler failed' });
    t.is(getCurrentRunTree(), outside);
  });
  for (const [index, collector] of collectors.entries()) {
    t.is(collector.records.length, 1);
    t.is(collector.records[0]!.project_name, `project-${index}`);
    t.falsy(collector.records[0]!.parent_run_id);
  }
  const after = new RunTree({ name: 'after' });
  t.is(after.client, outside.client);
  t.is(after.project_name, outside.project_name);
  t.is(after.address, outside.address);
  t.is(getCurrentRunTree(true), undefined);
});
