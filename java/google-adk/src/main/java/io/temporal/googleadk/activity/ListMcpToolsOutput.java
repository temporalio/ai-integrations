package io.temporal.googleadk.activity;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.genai.types.FunctionDeclaration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Output of the {@code googleadk.ListMcpTools} activity: the full declaration (name, description,
 * and parameter schema) of every tool the toolset currently offers.
 */
public record ListMcpToolsOutput(@JsonProperty("tools") List<FunctionDeclaration> tools) {

  /** Creates an output. */
  @JsonCreator
  public ListMcpToolsOutput {
    tools = tools == null ? List.of() : Collections.unmodifiableList(new ArrayList<>(tools));
  }
}
