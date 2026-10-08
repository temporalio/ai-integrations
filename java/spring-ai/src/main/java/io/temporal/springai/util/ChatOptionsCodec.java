package io.temporal.springai.util;

import java.util.Collection;
import java.util.function.Function;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.util.JacksonUtils;
import tools.jackson.core.JsonGenerator;
import tools.jackson.core.JsonParser;
import tools.jackson.databind.DatabindException;
import tools.jackson.databind.DeserializationContext;
import tools.jackson.databind.SerializationContext;
import tools.jackson.databind.ValueDeserializer;
import tools.jackson.databind.ValueSerializer;
import tools.jackson.databind.annotation.JsonPOJOBuilder;
import tools.jackson.databind.cfg.MapperConfig;
import tools.jackson.databind.introspect.Annotated;
import tools.jackson.databind.introspect.AnnotatedClass;
import tools.jackson.databind.introspect.AnnotatedMethod;
import tools.jackson.databind.introspect.JacksonAnnotationIntrospector;
import tools.jackson.databind.json.JsonMapper;

/** JSON mapper for immutable Spring AI options crossing an Activity boundary. */
public final class ChatOptionsCodec {
  private ChatOptionsCodec() {}

  /**
   * Uses each options type's public builder to reconstruct its concrete provider type. Tool
   * implementations stay in the workflow and cross the boundary as definitions only.
   */
  public static JsonMapper mapper() {
    return JacksonUtils.getDefaultJsonMapper()
        .rebuild()
        .annotationIntrospector(new OptionsIntrospector())
        .addMixIn(ToolCallingChatOptions.class, ToolOptionsMixin.class)
        .addMixIn(ToolCallingChatOptions.Builder.class, ToolOptionsMixin.class)
        .addMixIn(Function.class, FunctionMixin.class)
        .build();
  }

  @com.fasterxml.jackson.annotation.JsonIgnoreProperties(
      value = {"toolCallbacks", "toolNames", "toolContext"},
      ignoreUnknown = true)
  private abstract static class ToolOptionsMixin {}

  @com.fasterxml.jackson.annotation.JsonIgnoreType
  private abstract static class FunctionMixin {}

  // Native provider SDKs still use Jackson 2 annotations and codecs. Keep their wire values
  // intact while Spring AI's immutable option wrappers use the Jackson 3 mapper above.
  private static final com.fasterxml.jackson.databind.json.JsonMapper NATIVE_MAPPER =
      com.fasterxml.jackson.databind.json.JsonMapper.builder()
          .findAndAddModules()
          .disable(
              com.fasterxml.jackson.databind.MapperFeature.AUTO_DETECT_GETTERS,
              com.fasterxml.jackson.databind.MapperFeature.AUTO_DETECT_IS_GETTERS)
          .build();

  private static final class NativeOptionsSerializer extends ValueSerializer<Object> {
    @Override
    public void serialize(Object value, JsonGenerator generator, SerializationContext context) {
      try {
        generator.writeRawValue(NATIVE_MAPPER.writeValueAsString(value));
      } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
        throw DatabindException.from(generator, "Could not serialize native provider options", e);
      }
    }
  }

  private static final class NativeOptionsDeserializer extends ValueDeserializer<Object> {
    private final Class<?> type;

    private NativeOptionsDeserializer(Class<?> type) {
      this.type = type;
    }

    @Override
    public Object deserialize(JsonParser parser, DeserializationContext context) {
      try {
        return NATIVE_MAPPER.readValue(context.readTree(parser).toString(), type);
      } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
        throw DatabindException.from(parser, "Could not deserialize native provider options", e);
      }
    }
  }

  private static final class OptionsIntrospector extends JacksonAnnotationIntrospector {
    @Override
    public AnnotatedMethod resolveSetterConflict(
        MapperConfig<?> config, AnnotatedMethod first, AnnotatedMethod second) {
      // Provider builders often offer both a collection setter and an accumulating varargs
      // overload. JSON arrays should use the collection setter to replace the field once.
      if (first.getAnnotated().isVarArgs()
          && Collection.class.isAssignableFrom(second.getRawParameterType(0))) {
        return second;
      }
      if (second.getAnnotated().isVarArgs()
          && Collection.class.isAssignableFrom(first.getRawParameterType(0))) {
        return first;
      }
      return super.resolveSetterConflict(config, first, second);
    }

    @Override
    public Object findSerializer(MapperConfig<?> config, Annotated type) {
      Object serializer = super.findSerializer(config, type);
      if (serializer == null
          && type.hasAnnotation(com.fasterxml.jackson.databind.annotation.JsonSerialize.class)) {
        return new NativeOptionsSerializer();
      }
      return serializer;
    }

    @Override
    public Object findDeserializer(MapperConfig<?> config, Annotated type) {
      Object deserializer = super.findDeserializer(config, type);
      if (deserializer == null
          && type.hasAnnotation(com.fasterxml.jackson.databind.annotation.JsonDeserialize.class)) {
        return new NativeOptionsDeserializer(type.getRawType());
      }
      return deserializer;
    }

    @Override
    public Class<?> findPOJOBuilder(MapperConfig<?> config, AnnotatedClass type) {
      Class<?> annotatedBuilder = super.findPOJOBuilder(config, type);
      if (annotatedBuilder != null) {
        return annotatedBuilder;
      }
      if (!ChatOptions.class.isAssignableFrom(type.getRawType())
          && !type.getRawType().getName().startsWith("org.springframework.ai.")) {
        return null;
      }
      try {
        Class<?> builder = type.getRawType().getMethod("builder").invoke(null).getClass();
        if (ChatOptions.class.isAssignableFrom(type.getRawType())
            || type.getRawType().isAssignableFrom(builder.getMethod("build").getReturnType())) {
          return builder;
        }
      } catch (ReflectiveOperationException e) {
        // Custom bean-style options without a builder retain Jackson's normal handling.
      }
      return null;
    }

    @Override
    public JsonPOJOBuilder.Value findPOJOBuilderConfig(
        MapperConfig<?> config, AnnotatedClass type) {
      JsonPOJOBuilder.Value annotatedConfig = super.findPOJOBuilderConfig(config, type);
      if (annotatedConfig != null) {
        return annotatedConfig;
      }
      if (ChatOptions.Builder.class.isAssignableFrom(type.getRawType())
          || type.getRawType().getName().startsWith("org.springframework.ai.")) {
        return new JsonPOJOBuilder.Value("build", "");
      }
      return null;
    }
  }
}
