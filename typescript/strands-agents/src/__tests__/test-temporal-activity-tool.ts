import { ToolValidationError } from '@strands-agents/sdk';
import test from 'ava';
import { TemporalAgent, workflow } from '..';

test('activityAsTool without options can be registered with TemporalAgent', (t) => {
  const tool = workflow.activityAsTool('getWeather');
  const agent = new TemporalAgent({ model: 'test', printer: false, tools: [tool] });

  t.is(agent.toolRegistry.get('getWeather'), tool);
  t.true(tool.description === undefined);
  t.false(Object.hasOwn(tool, 'description'));
  t.false(Object.hasOwn(tool.toolSpec, 'description'));
});

test('activityAsTool omits an unspecified description, including with a name override', (t) => {
  for (const options of [{}, { description: undefined }, { name: 'weather' }]) {
    const tool = workflow.activityAsTool('getWeather', options);
    const agent = new TemporalAgent({ model: 'test', printer: false, tools: [tool] });

    t.is(agent.toolRegistry.get(options.name ?? 'getWeather'), tool);
    t.true(tool.description === undefined);
    t.false(Object.hasOwn(tool, 'description'));
    t.false(Object.hasOwn(tool.toolSpec, 'description'));
  }
});

test('activityAsTool preserves an explicit description', (t) => {
  const tool = workflow.activityAsTool('getWeather', { name: 'weather', description: 'Get the weather for a city' });
  const agent = new TemporalAgent({ model: 'test', printer: false, tools: [tool] });

  t.is(agent.toolRegistry.get('weather'), tool);
  t.is(tool.description, 'Get the weather for a city');
  t.is(tool.toolSpec.description, tool.description);
});

test('activityAsTool preserves Strands validation of an explicitly empty description', (t) => {
  const tool = workflow.activityAsTool('getWeather', { description: '' });

  t.is(tool.description, '');
  t.is(tool.toolSpec.description, '');
  t.throws(() => new TemporalAgent({ model: 'test', printer: false, tools: [tool] }), {
    instanceOf: ToolValidationError,
    message: 'Tool description must be a non-empty string',
  });
});
