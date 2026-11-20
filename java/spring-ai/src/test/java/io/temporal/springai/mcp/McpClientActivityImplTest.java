package io.temporal.springai.mcp;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import io.modelcontextprotocol.client.McpSyncClient;
import io.modelcontextprotocol.spec.McpError;
import io.modelcontextprotocol.spec.McpSchema;
import io.temporal.failure.ApplicationFailure;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class McpClientActivityImplTest {

  private static final McpSchema.CallToolRequest REQUEST =
      new McpSchema.CallToolRequest("tool", java.util.Map.of());

  static Stream<Integer> nonRetryableCodes() {
    return Stream.of(
        McpSchema.ErrorCodes.PARSE_ERROR,
        McpSchema.ErrorCodes.INVALID_REQUEST,
        McpSchema.ErrorCodes.METHOD_NOT_FOUND,
        McpSchema.ErrorCodes.INVALID_PARAMS);
  }

  @ParameterizedTest
  @MethodSource("nonRetryableCodes")
  void callToolProtocolErrorIsNonRetryable(int code) {
    McpSyncClient client = client("srv");
    McpError error = protocolError(code);
    when(client.callTool(any())).thenThrow(error);

    ApplicationFailure failure =
        assertThrows(
            ApplicationFailure.class,
            () -> new McpClientActivityImpl(List.of(client)).callTool("srv", REQUEST));

    assertTrue(failure.isNonRetryable());
    assertEquals("McpError", failure.getType());
    assertSame(error, failure.getCause());
  }

  @ParameterizedTest
  @MethodSource("nonRetryableCodes")
  void listToolsProtocolErrorIsNonRetryable(int code) {
    McpSyncClient client = client("srv");
    when(client.listTools()).thenThrow(protocolError(code));

    ApplicationFailure failure =
        assertThrows(
            ApplicationFailure.class, () -> new McpClientActivityImpl(List.of(client)).listTools());

    assertTrue(failure.isNonRetryable());
  }

  @Test
  void protocolErrorBehindWrapperIsNonRetryable() {
    McpSyncClient client = client("srv");
    when(client.callTool(any()))
        .thenThrow(
            new RuntimeException("wrapped", protocolError(McpSchema.ErrorCodes.INVALID_PARAMS)));

    ApplicationFailure failure =
        assertThrows(
            ApplicationFailure.class,
            () -> new McpClientActivityImpl(List.of(client)).callTool("srv", REQUEST));

    assertTrue(failure.isNonRetryable());
  }

  @Test
  void internalErrorAndTransportErrorsStayRetryable() {
    McpSyncClient client = client("srv");
    when(client.callTool(any()))
        .thenThrow(protocolError(McpSchema.ErrorCodes.INTERNAL_ERROR))
        .thenThrow(new IllegalStateException("connection lost"));
    McpClientActivityImpl activity = new McpClientActivityImpl(List.of(client));

    assertThrows(McpError.class, () -> activity.callTool("srv", REQUEST));
    assertThrows(IllegalStateException.class, () -> activity.callTool("srv", REQUEST));
  }

  private static McpError protocolError(int code) {
    return new McpError(new McpSchema.JSONRPCResponse.JSONRPCError(code, "rejected", null));
  }

  private static McpSyncClient client(String name) {
    McpSyncClient client = mock(McpSyncClient.class);
    when(client.getClientInfo())
        .thenReturn(McpSchema.Implementation.builder(name, "1.0.0").build());
    return client;
  }
}
