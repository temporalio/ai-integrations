package io.temporal.googleadk.activity;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

/** Input of the {@code googleadk.ListMcpTools} activity: the registered toolset name. */
public record ListMcpToolsInput(@JsonProperty("toolset") String toolset) {

  /** Creates an input. */
  @JsonCreator
  public ListMcpToolsInput {}
}
