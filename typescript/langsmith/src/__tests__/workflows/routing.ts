import { address } from 'langsmith';
import type { RunTreeConfig } from 'langsmith/run_trees';
import { traceable } from 'langsmith/traceable';
import { createNexusServiceClient, executeChild, proxyActivities, workflowInfo } from '@temporalio/workflow';
import type * as activities from '../activities/routing';
import { comprehensiveNexusService } from '../stubs/nexus';

export const ROUTING_ENDPOINT = 'langsmith-routing-endpoint';
export const ROUTING_ADDRESS = address.agent('support', 'staging');

const { routingActivity } = proxyActivities<typeof activities>({ startToCloseTimeout: '30 seconds' });

export async function RoutingChildWorkflow(): Promise<string> {
  return traceable(async () => routingActivity('child'), { name: 'routing-child' })();
}

export async function RoutingWorkflow(source: string): Promise<string> {
  const step = async (): Promise<string> => {
    await routingActivity('parent');
    await executeChild(RoutingChildWorkflow, { workflowId: `${workflowInfo().workflowId}-child` });
    await createNexusServiceClient({ service: comprehensiveNexusService, endpoint: ROUTING_ENDPOINT })
      .executeOperation(comprehensiveNexusService.operations.greet, { name: 'routing' }, {});
    return 'done';
  };
  if (source === 'runtime') {
    return traceable(async (_config: Partial<RunTreeConfig>) => step(), {
      name: 'routing-step',
      argsConfigPath: [0],
    })({ address: ROUTING_ADDRESS });
  }
  return traceable(step, {
    name: 'routing-step',
    ...(source === 'decorator' ? { address: ROUTING_ADDRESS } : {}),
  })();
}
