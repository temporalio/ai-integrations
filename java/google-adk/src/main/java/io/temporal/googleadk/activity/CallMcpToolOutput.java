package io.temporal.googleadk.activity;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/** Output of the {@code googleadk.CallMcpTool} activity: the tool result as ADK presents it. */
public record CallMcpToolOutput(@JsonProperty("result") Map<String, Object> result) {

  /** Creates an output. */
  @JsonCreator
  public CallMcpToolOutput {
    result = result == null ? Map.of() : Collections.unmodifiableMap(new LinkedHashMap<>(result));
  }
}
