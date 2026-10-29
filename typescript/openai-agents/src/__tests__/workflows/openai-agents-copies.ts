import { Agent, Handoff, tool, type Model } from '@openai/agents-core';
import { ApplicationFailure } from '@temporalio/workflow';
import { TemporalOpenAIRunner } from '../../workflow';

// A second copy of @openai/agents-core, as a dependency that ships its own copy would bring. The
// test bundles this Workflow with `agents-core-copy` resolving to a copy of the package.
// eslint-disable-next-line @typescript-eslint/no-require-imports
const copy = require('agents-core-copy') as typeof import('@openai/agents-core');

export type AgentCopyCase =
  'agent-start-model-object' | 'handoff-callback-model-object' | 'tool-model-object' | 'string-model';

/**
 * Runs agents from the second copy of @openai/agents-core, whose `Agent.prototype` the plugin didn't
 * patch when it loaded. The `*-model-object` cases give such an agent a `Model` object after the check
 * before the run: an `agent_start` listener, a custom `Handoff`'s callback, or a tool between two
 * turns. `string-model` is the supported control.
 * Returns the run's failure or its output.
 */
export async function agentCopyWorkflow(kind: AgentCopyCase): Promise<string> {
  if ((copy.Agent as unknown) === Agent) throw ApplicationFailure.nonRetryable('agents-core-copy is not a second copy');
  const inlineModel: Model = {
    async getResponse() {
      throw new Error('the inline model ran in the Workflow');
    },
    // eslint-disable-next-line require-yield
    async *getStreamedResponse() {
      throw new Error('the inline model ran in the Workflow');
    },
  };
  let lead: Agent<any, any>;
  if (kind === 'agent-start-model-object') {
    const copied = new copy.Agent({ name: 'Lead', instructions: 'Answer.', model: 'lead-model' });
    copied.on('agent_start', () => (copied.model = inlineModel));
    lead = copied as unknown as Agent<any, any>;
  } else if (kind === 'handoff-callback-model-object') {
    const declared = new Agent({ name: 'Declared', instructions: 'Answer as declared.', model: 'declared-model' });
    const picked = new copy.Agent({ name: 'Picked', model: inlineModel });
    const desk = new Handoff(declared, async () => picked as unknown as Agent<any, any>);
    desk.toolName = 'transfer_to_desk';
    lead = new Agent({ name: 'Lead', instructions: 'Route.', model: 'lead-model', handoffs: [desk] });
  } else if (kind === 'tool-model-object') {
    const copied = new copy.Agent({ name: 'Lead', instructions: 'Switch.', model: 'lead-model' });
    const switchModel = tool({
      name: 'switch_model',
      description: '',
      parameters: { type: 'object' as const, properties: {}, required: [], additionalProperties: false as const },
      strict: true,
      execute: async () => {
        copied.model = inlineModel;
        return 'switched';
      },
    });
    copied.tools = [switchModel as any];
    lead = copied as unknown as Agent<any, any>;
  } else {
    lead = new copy.Agent({ name: 'Lead', instructions: 'Answer.', model: 'lead-model' }) as unknown as Agent<any, any>;
  }
  try {
    return (await new TemporalOpenAIRunner().run(lead, 'go')).finalOutput ?? '';
  } catch (err: unknown) {
    if (err instanceof ApplicationFailure) return `${err.type}: ${err.message}`;
    // Return other errors too, such as the inline model's, so the test sees them.
    if (err instanceof Error) return `${err.name}: ${err.message}`;
    throw err;
  }
}
