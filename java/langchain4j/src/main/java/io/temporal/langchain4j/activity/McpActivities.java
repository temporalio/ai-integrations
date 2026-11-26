package io.temporal.langchain4j.activity;

import io.temporal.activity.ActivityInterface;
import io.temporal.activity.ActivityMethod;

/**
 * Activities that list and call the tools of a LangChain4j {@code McpClient} registered on the
 * worker. {@code TemporalMcpClient} schedules them; workflows don't call them directly.
 */
@ActivityInterface
public interface McpActivities {

  /** The activity type name of {@link #listMcpTools(ListMcpToolsInput)}. */
  String LIST_MCP_TOOLS = "langchain4j.ListMcpTools";

  /** The activity type name of {@link #callMcpTool(CallMcpToolInput)}. */
  String CALL_MCP_TOOL = "langchain4j.CallMcpTool";

  /** Lists the tools of the MCP client registered under {@link ListMcpToolsInput#client()}. */
  @ActivityMethod(name = LIST_MCP_TOOLS)
  ListMcpToolsOutput listMcpTools(ListMcpToolsInput input);

  /** Calls a tool of the MCP client registered under {@link CallMcpToolInput#client()}. */
  @ActivityMethod(name = CALL_MCP_TOOL)
  CallMcpToolOutput callMcpTool(CallMcpToolInput input);
}
