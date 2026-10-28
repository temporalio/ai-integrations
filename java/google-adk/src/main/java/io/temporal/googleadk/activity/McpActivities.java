package io.temporal.googleadk.activity;

import io.temporal.activity.ActivityInterface;
import io.temporal.activity.ActivityMethod;

/**
 * The activities that serve MCP toolsets on a worker. Registered by the plugin; toolsets are
 * created on the worker by the factories configured on {@code GoogleAdkPlugin}.
 */
@ActivityInterface(namePrefix = "googleadk.")
public interface McpActivities {

  /** The activity type name of {@link #listMcpTools(ListMcpToolsInput)}. */
  String LIST_MCP_TOOLS = "googleadk.ListMcpTools";

  /** The activity type name of {@link #callMcpTool(CallMcpToolInput)}. */
  String CALL_MCP_TOOL = "googleadk.CallMcpTool";

  /** Lists the tools of a registered toolset. */
  @ActivityMethod
  ListMcpToolsOutput listMcpTools(ListMcpToolsInput input);

  /** Calls a tool of a registered toolset. */
  @ActivityMethod
  CallMcpToolOutput callMcpTool(CallMcpToolInput input);
}
