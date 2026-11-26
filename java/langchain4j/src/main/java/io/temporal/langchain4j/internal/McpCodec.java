package io.temporal.langchain4j.internal;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.ChatMessageDeserializer;
import dev.langchain4j.data.message.ChatMessageSerializer;
import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.service.tool.ToolExecutionResult;
import io.temporal.langchain4j.activity.CallMcpToolInput;
import io.temporal.langchain4j.activity.CallMcpToolOutput;
import java.io.UncheckedIOException;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Converts LangChain4j tool execution requests and results to and from the MCP activity's input and
 * output. Result contents use LangChain4j's own message JSON, so images and other content types
 * cross the activity boundary unchanged; the structured result and the attributes travel as JSON.
 */
public final class McpCodec {

  private static final ObjectMapper JSON = new ObjectMapper();
  private static final TypeReference<Map<String, Object>> MAP = new TypeReference<>() {};

  private McpCodec() {}

  /** Encodes a tool execution request for the client registered under {@code client}. */
  public static CallMcpToolInput encode(String client, ToolExecutionRequest request) {
    return new CallMcpToolInput(client, request.id(), request.name(), request.arguments());
  }

  /** Decodes the activity input back into the request it was encoded from. */
  public static ToolExecutionRequest decode(CallMcpToolInput input) {
    return ToolExecutionRequest.builder()
        .id(input.id())
        .name(input.tool())
        .arguments(input.arguments())
        .build();
  }

  /** Encodes a tool execution result as the activity result. */
  public static CallMcpToolOutput encode(String tool, ToolExecutionResult result) {
    List<Content> contents = result.resultContents();
    return new CallMcpToolOutput(
        contents.isEmpty()
            ? null
            : ChatMessageSerializer.messageToJson(
                ToolExecutionResultMessage.builder()
                    .id(tool)
                    .toolName(tool)
                    .contents(contents)
                    .build()),
        result.isError(),
        result.result() == null ? null : toJson(result.result()),
        result.attributes().isEmpty() ? null : toJson(result.attributes()),
        null);
  }

  /** Encodes the text of a failure that LangChain4j shows the model. */
  public static CallMcpToolOutput errorForLlm(String text) {
    return new CallMcpToolOutput(null, true, null, null, text);
  }

  /** Decodes the activity result back into a tool execution result. */
  public static ToolExecutionResult decode(CallMcpToolOutput output) {
    ToolExecutionResult.Builder result =
        ToolExecutionResult.builder().isError(output.isError()).result(fromJson(output.result()));
    if (output.contents() == null) {
      // A tool that returned no content at all; resultText() is then null, as on the worker.
      result.resultTextSupplier(() -> null);
    } else {
      result.resultContents(
          ((ToolExecutionResultMessage) ChatMessageDeserializer.messageFromJson(output.contents()))
              .contents());
    }
    if (output.attributes() != null) {
      result.attributes(fromJson(output.attributes(), MAP));
    }
    return result.build();
  }

  private static String toJson(Object value) {
    try {
      return JSON.writeValueAsString(value);
    } catch (JsonProcessingException e) {
      throw new IllegalArgumentException(
          "An MCP tool result must be JSON-serializable: " + e.getOriginalMessage(), e);
    }
  }

  @Nullable
  private static Object fromJson(@Nullable String json) {
    return json == null ? null : fromJson(json, new TypeReference<Object>() {});
  }

  private static <T> T fromJson(String json, TypeReference<T> type) {
    try {
      return JSON.readValue(json, type);
    } catch (JsonProcessingException e) {
      throw new UncheckedIOException(e);
    }
  }
}
