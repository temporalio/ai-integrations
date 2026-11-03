import { readFileSync } from 'node:fs';
import path from 'node:path';

import test from 'ava';
import type { NativeConnection, WorkerOptions } from '@temporalio/worker';
import { StatelessMCPServerProvider } from '../worker/mcp-provider';
import { StatefulMCPServerProvider } from '../worker/stateful-mcp-provider';
import { OpenAIAgentsPlugin } from '../worker/plugin';
import { FakeModelProvider } from './stubs/openai-agents';

type Activities = Record<string, unknown>;

const callerActivity = async () => 'caller';

const tail =
  'or remove the other Activity, run the other integration on its own Worker and task queue, or use one ' +
  'OpenAIAgentsPlugin instance on this Worker.';

test('configureWorker refuses to replace a different Activity with the same name', (t) => {
  const options = {
    modelProvider: new FakeModelProvider([]),
    mcpServerProviders: [
      new StatelessMCPServerProvider('docs', () => {
        throw new Error('not called');
      }),
      new StatefulMCPServerProvider(
        'notes',
        () => {
          throw new Error('not called');
        },
        {} as NativeConnection
      ),
    ],
  };
  const plugin = new OpenAIAgentsPlugin(options);

  const refusals: Record<string, string> = {
    invokeModelActivity:
      "Duplicate registration of Activity 'invokeModelActivity': OpenAIAgentsPlugin would replace a different " +
      `Activity with this name. Rename ${tail}`,
    'docs-stateless-call-tool':
      "Duplicate registration of Activity 'docs-stateless-call-tool': OpenAIAgentsPlugin provider 'docs' would " +
      `replace a different Activity with this name. Rename the provider, rename ${tail}`,
    // The name the caller passed, without the stateful provider's internal suffix.
    'notes-stateful-server-session':
      "Duplicate registration of Activity 'notes-stateful-server-session': OpenAIAgentsPlugin provider 'notes' " +
      `would replace a different Activity with this name. Rename the provider, rename ${tail}`,
  };
  for (const [name, message] of Object.entries(refusals)) {
    t.throws(
      () => plugin.configureWorker({ taskQueue: 'test', activities: { [name]: callerActivity } } as WorkerOptions),
      { instanceOf: TypeError, message }
    );
  }

  // Caller Activities with other names stay beside the plugin's.
  const once = plugin.configureWorker({ taskQueue: 'test', activities: { lookUp: callerActivity } } as WorkerOptions);
  t.is((once.activities as Activities).lookUp, callerActivity);

  // Worker.create runs configureWorker twice when one instance is on both the NativeConnection
  // and the Worker.
  const twice = plugin.configureWorker(once);
  t.is((twice.activities as Activities).invokeModelActivity, (once.activities as Activities).invokeModelActivity);

  // A second instance built from the same options is the same plugin.
  const same = new OpenAIAgentsPlugin(options).configureWorker(once);
  t.is((same.activities as Activities).lookUp, callerActivity);

  // A second instance with another model provider would replace the first one's Activities.
  t.throws(() => new OpenAIAgentsPlugin({ modelProvider: new FakeModelProvider([]) }).configureWorker(once), {
    instanceOf: TypeError,
    message:
      "Duplicate registration of Activity 'invokeModelActivity': OpenAIAgentsPlugin would replace a different " +
      `Activity with this name. Rename ${tail}`,
  });
});

test('reports the package version for worker heartbeats', (t) => {
  t.is(new OpenAIAgentsPlugin({ modelProvider: new FakeModelProvider([]) }).version, JSON.parse(readFileSync(path.join(__dirname, '../../package.json'), 'utf8')).version);
});
