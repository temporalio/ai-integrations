import { ApplicationFailure, DefaultFailureConverter } from '@temporalio/common';
import type { FailureConverter, PayloadConverter, ProtoFailure, SerializationContext } from '@temporalio/common';

// A caller's failure converter: failure messages and stack traces go into an encoded payload, and a
// failure of type `Custom` is decoded with its own message, so a test can tell who decoded it. It
// keeps its inner converter in a private field and delegates to it, so the inner converter encodes
// each cause, and the module exports it frozen; the plugin must work with all three. It also wraps a
// failure of type `Business` in an `Envelope` failure, a level the error does not have, and unwraps
// it when decoding; the plugin must keep that level and what it wraps.
class EncodingFailureConverter implements FailureConverter {
  readonly #inner = new DefaultFailureConverter({ encodeCommonAttributes: true });

  errorToFailure(err: unknown, payloadConverter: PayloadConverter, context?: SerializationContext): ProtoFailure {
    const failure = this.#inner.errorToFailure(err, payloadConverter, context);
    if (!(err instanceof ApplicationFailure && err.type === 'Business')) return failure;
    const envelope = ApplicationFailure.create({ message: 'envelope', type: 'Envelope' });
    return { ...this.#inner.errorToFailure(envelope, payloadConverter, context), cause: failure };
  }

  failureToError(failure: ProtoFailure, payloadConverter: PayloadConverter, context?: SerializationContext): Error {
    if (failure.applicationFailureInfo?.type === 'Envelope' && failure.cause) {
      return this.failureToError(failure.cause, payloadConverter, context);
    }
    const err = this.#inner.failureToError(failure, payloadConverter, context);
    return err instanceof ApplicationFailure && err.type === 'Custom'
      ? ApplicationFailure.create({ message: 'decoded by the caller', type: 'Custom' })
      : err;
  }
}

const converter = new EncodingFailureConverter();

// ts-prune-ignore-next
export const failureConverter: FailureConverter = Object.freeze({
  errorToFailure: (err: unknown, payloadConverter: PayloadConverter, context?: SerializationContext) =>
    converter.errorToFailure(err, payloadConverter, context),
  failureToError: (failure: ProtoFailure, payloadConverter: PayloadConverter, context?: SerializationContext) =>
    converter.failureToError(failure, payloadConverter, context),
});
