/** Not a workflow, just interceptors */

import { ApplicationFailure } from '@temporalio/common';
import type { WorkflowInterceptors } from '@temporalio/workflow';
import { findPayloadChange } from '../helpers/payload-round-trip';

// The Workflow converts Activity input inside the sandbox, so the check runs here too.
export const interceptors = (): WorkflowInterceptors => ({
  outbound: [
    {
      scheduleActivity(input, next) {
        const change = findPayloadChange(input.args);
        if (change !== undefined) {
          const message = `${change.path}: ${change.sent} became ${change.received}`;
          throw ApplicationFailure.nonRetryable(message, 'PayloadChange');
        }
        return next(input);
      },
    },
  ],
});
