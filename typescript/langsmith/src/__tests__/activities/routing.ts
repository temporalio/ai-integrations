import { traceable } from 'langsmith/traceable';

export const routingActivity = traceable(async (input: string): Promise<string> => input, {
  name: 'routing-activity',
});
