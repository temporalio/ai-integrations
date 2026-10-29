// The README quick-start Workflow as an ES module. Tests bundle this file itself, so the bundler
// sees `import` syntax, as it does for a `.ts` Workflow file run with ts-node or tsx.
import { Agent } from '@openai/agents-core';
import { TemporalOpenAIRunner } from '@temporalio/openai-agents/workflow';

export async function esmModuleAgentWorkflow(prompt) {
  const agent = new Agent({
    name: 'Assistant',
    instructions: 'You only respond in haikus.',
    model: 'gpt-4o-mini',
  });
  const result = await new TemporalOpenAIRunner().run(agent, prompt);
  return result.finalOutput ?? '';
}
