package io.temporal.googleadk.converter;

import io.temporal.common.converter.DataConverter;
import io.temporal.common.converter.DefaultDataConverter;
import io.temporal.common.converter.GlobalDataConverter;
import java.util.Objects;
import javax.annotation.Nullable;

/** Factory for Temporal data converters that understand Google ADK and GenAI types. */
public final class GoogleAdkDataConverter {

  private GoogleAdkDataConverter() {}

  /**
   * Returns Temporal's default data converter with its JSON converter replaced by {@link
   * GoogleAdkPayloadConverter}.
   */
  public static DefaultDataConverter newDefaultInstance() {
    return DefaultDataConverter.newDefaultInstance()
        .withPayloadConverterOverrides(new GoogleAdkPayloadConverter());
  }

  /**
   * Replaces the JSON converter of the given default data converter with {@link
   * GoogleAdkPayloadConverter}; its other payload converters and its failure converter are kept.
   * The SDK offers no way to copy a {@link DefaultDataConverter}, so the converter is modified in
   * place and returned: do not share it with clients that should not see ADK types. {@link
   * DefaultDataConverter#STANDARD_INSTANCE}, which the SDK uses internally, is never modified; a
   * new default converter with ADK support is returned for it.
   */
  public static DefaultDataConverter withAdkTypes(DefaultDataConverter converter) {
    Objects.requireNonNull(converter, "converter");
    if (converter == DefaultDataConverter.STANDARD_INSTANCE) {
      return newDefaultInstance();
    }
    return converter.withPayloadConverterOverrides(new GoogleAdkPayloadConverter());
  }

  /**
   * Returns a data converter that understands ADK types, derived from the given one. A null
   * converter means the client has not configured one, so the global converter is used instead. The
   * SDK's own {@link DefaultDataConverter#STANDARD_INSTANCE} is never modified: a new default
   * converter with ADK support is returned for it. Any other {@link DefaultDataConverter}, whether
   * set on the client or registered with {@link GlobalDataConverter#register}, belongs to the
   * application and is extended in place, as described in {@link
   * #withAdkTypes(DefaultDataConverter)}.
   *
   * @throws IllegalStateException if the converter is not a {@link DefaultDataConverter}; wrap such
   *     converters around {@link #newDefaultInstance()} yourself
   */
  public static DataConverter withAdkTypes(@Nullable DataConverter converter) {
    DataConverter base = converter != null ? converter : GlobalDataConverter.get();
    if (!(base instanceof DefaultDataConverter)) {
      throw new IllegalStateException(
          "Cannot add Google ADK payload support to a "
              + base.getClass().getName()
              + ". Build your data converter around GoogleAdkDataConverter.newDefaultInstance()"
              + " (for example new CodecDataConverter(GoogleAdkDataConverter.newDefaultInstance(),"
              + " codecs)) and disable the plugin's data converter customization.");
    }
    return withAdkTypes((DefaultDataConverter) base);
  }
}
