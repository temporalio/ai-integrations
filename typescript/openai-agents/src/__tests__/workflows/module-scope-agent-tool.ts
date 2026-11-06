import { Agent, type Tool } from '@openai/agents-core';

// The Workflow module imports this before `@temporalio/openai-agents/workflow`, so the tool is made
// before that module loads. The plugin's Workflow interceptor module, which loads first, records it.
export const moduleScopeTool: Tool = new Agent({
  name: 'Early',
  instructions: 'Answer as early.',
  model: 'early-model',
}).asTool({ toolName: 'ask_early' });
