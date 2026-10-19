package io.temporal.googleadk.activity;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/** Input of the {@code googleadk.CallMcpTool} activity. */
public record CallMcpToolInput(
    @JsonProperty("toolset") String toolset,
    @JsonProperty("tool") String tool,
    @JsonProperty("args") Map<String, Object> args) {

  /** Creates an input. */
  @JsonCreator
  public CallMcpToolInput {
    // Values may be null: tool arguments and results are JSON, which allows null.
    args = args == null ? Map.of() : Collections.unmodifiableMap(new LinkedHashMap<>(args));
  }
}
