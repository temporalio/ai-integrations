package io.temporal.googleadk.converter;

import io.temporal.common.converter.DataConverter;
import io.temporal.common.converter.DefaultDataConverter;
import io.temporal.common.converter.GlobalDataConverter;
import javax.annotation.Nullable;

/** Factory for Temporal data converters that understand Google ADK and GenAI types. */
public final class GoogleAdkDataConverter {

  private GoogleAdkDataConverter() {}

  /**
   * Returns Temporal's default data converter with its JSON converter replaced by {@link
   * GoogleAdkPayloadConverter}.
   */
  public static DefaultDataConverter newDefaultInstance() {
    return withAdkTypes(DefaultDataConverter.newDefaultInstance());
  }

  /**
   * Returns a copy of the given default data converter whose JSON converter is replaced by {@link
   * GoogleAdkPayloadConverter}. Other payload converters and the failure converter are kept.
   */
  public static DefaultDataConverter withAdkTypes(DefaultDataConverter converter) {
    return converter.withPayloadConverterOverrides(new GoogleAdkPayloadConverter());
  }

  /**
   * Returns a data converter that understands ADK types, derived from the given one when it is a
   * {@link DefaultDataConverter}. A null converter means the client has not configured one, so the
   * global default is extended instead.
   *
   * @throws IllegalStateException if the converter is not a {@link DefaultDataConverter}; wrap such
   *     converters around {@link #newDefaultInstance()} yourself
   */
  public static DataConverter withAdkTypes(@Nullable DataConverter converter) {
    DataConverter base = converter != null ? converter : GlobalDataConverter.get();
    if (base instanceof DefaultDataConverter) {
      return withAdkTypes((DefaultDataConverter) base);
    }
    throw new IllegalStateException(
        "Cannot add Google ADK payload support to a "
            + base.getClass().getName()
            + ". Build your data converter around GoogleAdkDataConverter.newDefaultInstance()"
            + " (for example new CodecDataConverter(GoogleAdkDataConverter.newDefaultInstance(),"
            + " codecs)) and disable the plugin's data converter customization.");
  }
}
