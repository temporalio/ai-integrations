package io.temporal.googleadk.converter;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.adk.JsonBaseModel;
import com.google.genai.JsonSerializable;
import io.temporal.api.common.v1.Payload;
import io.temporal.common.converter.DataConverterException;
import io.temporal.common.converter.JacksonJsonPayloadConverter;
import io.temporal.common.converter.PayloadConverter;
import java.lang.reflect.Type;
import java.util.Objects;
import java.util.Optional;

/**
 * The {@code json/plain} payload converter installed by the plugin: Temporal's default Jackson
 * configuration plus the settings Google ADK's own mapper relies on.
 *
 * <p>ADK's request, response, event, and session types are AutoValue classes that omit absent
 * optionals, ignore unknown properties, and keep a few live members (such as the tool instances on
 * a request) out of JSON. Temporal's default converter writes every absent optional as {@code null}
 * and serializes a request's live tool instances, which it cannot read back. This converter uses
 * one mapper for everything, so a workflow input, result, signal, or query may freely mix ADK types
 * with your own: Temporal's defaults (field visibility, ISO-8601 dates, the Java time and {@code
 * Optional} modules) stay in place, absent optionals are omitted, unknown properties are ignored,
 * and a request's transient members are skipped.
 */
public final class GoogleAdkPayloadConverter implements PayloadConverter {

  private static final String JSON_ENCODING = "json/plain";

  private final JacksonJsonPayloadConverter delegate;

  /** Creates a converter from Temporal's default mapper configuration. */
  public GoogleAdkPayloadConverter() {
    this.delegate = new JacksonJsonPayloadConverter(newObjectMapper());
  }

  /**
   * Creates a converter from a copy of the given mapper with the ADK settings applied, for
   * applications that register their own Jackson modules or features.
   */
  public GoogleAdkPayloadConverter(ObjectMapper baseMapper) {
    this.delegate =
        new JacksonJsonPayloadConverter(configure(Objects.requireNonNull(baseMapper).copy()));
  }

  /** Returns Temporal's default mapper configuration with the ADK settings applied. */
  public static ObjectMapper newObjectMapper() {
    return configure(JacksonJsonPayloadConverter.newDefaultObjectMapper());
  }

  private static ObjectMapper configure(ObjectMapper mapper) {
    mapper.addMixIn(JsonBaseModel.class, AdkTypeMixin.class);
    mapper.addMixIn(JsonSerializable.class, AdkTypeMixin.class);
    return mapper;
  }

  @Override
  public String getEncodingType() {
    return JSON_ENCODING;
  }

  @Override
  public Optional<Payload> toData(Object value) throws DataConverterException {
    return delegate.toData(value);
  }

  @Override
  public <T> T fromData(Payload content, Class<T> valueClass, Type valueType)
      throws DataConverterException {
    return delegate.fromData(content, valueClass, valueType);
  }

  /**
   * Gives ADK and GenAI types the JSON shape their own mappers produce, without touching other
   * types: only annotated accessors are read (Temporal's field visibility would otherwise serialize
   * the backing fields of their generated subclasses, including members such as a request's live
   * tool instances that ADK keeps out of JSON), absent optionals are omitted, and unknown
   * properties are ignored.
   */
  @JsonAutoDetect(fieldVisibility = JsonAutoDetect.Visibility.NONE)
  @JsonInclude(value = JsonInclude.Include.NON_ABSENT, content = JsonInclude.Include.NON_ABSENT)
  @JsonIgnoreProperties(ignoreUnknown = true)
  private abstract static class AdkTypeMixin {}
}
