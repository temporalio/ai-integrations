import test from 'ava';
import type { WorkerOptions } from '@temporalio/worker';
import { StatelessMCPServerProvider } from '../worker/mcp-provider';
import { OpenAIAgentsPlugin } from '../worker/plugin';
import { FakeModelProvider } from './stubs/openai-agents';

type Activities = Record<string, unknown>;

const callerActivity = async () => 'caller';

test('configureWorker refuses to replace a caller Activity, but not its own', (t) => {
  const plugin = new OpenAIAgentsPlugin({
    modelProvider: new FakeModelProvider([]),
    mcpServerProviders: [
      new StatelessMCPServerProvider('docs', () => {
        throw new Error('not called');
      }),
    ],
  });

  for (const name of ['invokeModelActivity', 'docs-stateless-call-tool']) {
    t.throws(
      () => plugin.configureWorker({ taskQueue: 'test', activities: { [name]: callerActivity } } as WorkerOptions),
      {
        instanceOf: TypeError,
        message:
          `Duplicate registration of Activity '${name}': OpenAIAgentsPlugin would replace a different Activity ` +
          'with this name. Rename or remove the other registration.',
      }
    );
  }

  // Caller Activities with other names stay beside the plugin's.
  const once = plugin.configureWorker({ taskQueue: 'test', activities: { lookUp: callerActivity } } as WorkerOptions);
  t.is((once.activities as Activities).lookUp, callerActivity);

  // Worker.create runs configureWorker twice when one instance is on both the NativeConnection
  // and the Worker.
  const twice = plugin.configureWorker(once);
  t.is((twice.activities as Activities).invokeModelActivity, (once.activities as Activities).invokeModelActivity);

  // Chained instances keep the last instance's Activities, as before.
  const chained = new OpenAIAgentsPlugin({ modelProvider: new FakeModelProvider([]) }).configureWorker(once);
  t.not((chained.activities as Activities).invokeModelActivity, (once.activities as Activities).invokeModelActivity);
  t.is((chained.activities as Activities).lookUp, callerActivity);
});
