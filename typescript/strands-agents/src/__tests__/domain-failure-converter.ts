import { DefaultFailureConverter, type FailureConverter, TemporalFailure } from '@temporalio/common';

class DomainFailure extends TemporalFailure {}

// A caller's failure converter that decodes application failures to a failure class of its own, so
// workflow code gets a `TemporalFailure` that is not an `ApplicationFailure`.
class DomainFailureConverter extends DefaultFailureConverter {
  override failureToErrorInner(...args: Parameters<DefaultFailureConverter['failureToErrorInner']>): Error {
    const err = super.failureToErrorInner(...args);
    return args[0].applicationFailureInfo ? new DomainFailure(err.message, err.cause as Error | undefined) : err;
  }
}

// ts-prune-ignore-next
export const failureConverter: FailureConverter = new DomainFailureConverter();
