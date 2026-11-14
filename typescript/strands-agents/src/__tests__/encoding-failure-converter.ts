import { DefaultFailureConverter, type FailureConverter } from '@temporalio/common';

// A caller's failure converter: failure messages and stack traces go into an encoded payload, as the
// Temporal docs recommend for sensitive error text.
// ts-prune-ignore-next
export const failureConverter: FailureConverter = new DefaultFailureConverter({ encodeCommonAttributes: true });
