/**
 * A failure converter that converts a copy of each failure it is given, as a custom
 * converter may: the errors mean the same, but their `failure` is not the object the
 * Workflow's activation carried.
 */

import {
  DefaultFailureConverter,
  type PayloadConverter,
  type ProtoFailure,
  type SerializationContext,
} from '@temporalio/common';

class CopyingFailureConverter extends DefaultFailureConverter {
  override failureToError(
    failure: ProtoFailure,
    payloadConverter: PayloadConverter,
    context?: SerializationContext
  ): Error {
    return super.failureToError({ ...failure }, payloadConverter, context);
  }
}

export const failureConverter = new CopyingFailureConverter();
