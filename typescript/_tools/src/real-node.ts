/** MCP child environments intentionally omit Bazel launcher variables. */
const pinnedNode = process.env.JS_BINARY__NODE_BINARY;
if (pinnedNode) {
  process.execPath = pinnedNode;
  process.argv[0] = pinnedNode;
}
