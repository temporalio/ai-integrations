import { TestWorkflowEnvironment } from '@temporalio/testing';

/** Connect to the dev server owned by the test launcher. */
export async function createTestEnvironment(): Promise<TestWorkflowEnvironment> {
  const address = process.env.TEMPORAL_TEST_ADDRESS;
  if (!address) {
    throw new Error('Run live Temporal tests with `make test` or set TEMPORAL_TEST_ADDRESS to a running server.');
  }
  return await TestWorkflowEnvironment.createFromExistingServer({ address });
}
