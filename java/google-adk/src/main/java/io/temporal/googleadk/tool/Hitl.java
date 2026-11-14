package io.temporal.googleadk.tool;

import com.google.adk.JsonBaseModel;
import com.google.adk.events.Event;
import com.google.adk.events.ToolConfirmation;
import com.google.adk.flows.llmflows.Functions;
import com.google.genai.types.Content;
import com.google.genai.types.FunctionCall;
import com.google.genai.types.FunctionResponse;
import com.google.genai.types.Part;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Human-in-the-loop helpers built on ADK's own long-running tool and tool-confirmation flow.
 *
 * <p>A tool created with {@code requireConfirmation} does not run until a user confirms it: the run
 * ends with an {@code adk_request_confirmation} call among its events. Find those calls with {@link
 * #pendingConfirmations(List)}, wait for the decision with a signal or an update, answer with
 * {@link #confirmationResponse(ConfirmationDecision...)}, and run the next turn with that content:
 * ADK then executes the confirmed tool and continues.
 *
 * <pre>{@code
 * List<Event> events = TemporalAdk.run(runner, userId, sessionId, question);
 * List<Hitl.PendingConfirmation> pending = Hitl.pendingConfirmations(events);
 * if (!pending.isEmpty()) {
 *   Workflow.await(() -> decision != null); // set by a signal handler
 *   events =
 *       TemporalAdk.run(
 *           runner,
 *           userId,
 *           sessionId,
 *           Hitl.confirmationResponse(
 *               new Hitl.ConfirmationDecision(pending.get(0).functionCallId(), decision)));
 * }
 * }</pre>
 *
 * <p>Long-running tools that return nothing pause the run the same way: the model's turn ends with
 * the call listed by {@link #longRunningCalls(List)}, and {@link #functionResponse(FunctionCall,
 * Map)} builds the content that resumes it.
 */
public final class Hitl {

  /** A tool call waiting for a user's decision. */
  public record PendingConfirmation(
      String functionCallId,
      FunctionCall originalFunctionCall,
      @Nullable String hint,
      @Nullable Object payload) {

    /** Creates a pending confirmation. */
    public PendingConfirmation {
      Objects.requireNonNull(functionCallId, "functionCallId");
      Objects.requireNonNull(originalFunctionCall, "originalFunctionCall");
    }

    /** Returns the name of the tool awaiting confirmation. */
    public String toolName() {
      return originalFunctionCall.name().orElse("");
    }
  }

  /** A user's decision about a pending confirmation, optionally carrying a payload. */
  public record ConfirmationDecision(
      String functionCallId, boolean confirmed, @Nullable Object payload) {

    /** Creates a decision without a payload. */
    public ConfirmationDecision(String functionCallId, boolean confirmed) {
      this(functionCallId, confirmed, null);
    }

    /** Creates a decision. */
    public ConfirmationDecision {
      Objects.requireNonNull(functionCallId, "functionCallId");
    }
  }

  private Hitl() {}

  /**
   * Returns the confirmation requests among the events, in the order ADK emitted them. The next
   * turn must answer every request of a turn, each with its own decision.
   */
  public static List<PendingConfirmation> pendingConfirmations(List<Event> events) {
    List<PendingConfirmation> pending = new ArrayList<>();
    for (Event event : events) {
      Set<String> longRunning = event.longRunningToolIds().orElse(Set.of());
      for (FunctionCall call : event.functionCalls()) {
        if (!Functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME.equals(call.name().orElse(null))
            || call.id().isEmpty()
            || !longRunning.contains(call.id().get())) {
          continue;
        }
        Map<String, Object> args = call.args().orElse(Map.of());
        Object originalCall = args.get("originalFunctionCall");
        if (originalCall == null) {
          throw new IllegalStateException(
              "Confirmation request " + call.id().get() + " carries no originalFunctionCall");
        }
        FunctionCall original =
            JsonBaseModel.getMapper().convertValue(originalCall, FunctionCall.class);
        ToolConfirmation confirmation =
            args.get("toolConfirmation") == null
                ? null
                : JsonBaseModel.getMapper()
                    .convertValue(args.get("toolConfirmation"), ToolConfirmation.class);
        pending.add(
            new PendingConfirmation(
                call.id().get(),
                original,
                confirmation == null ? null : confirmation.hint(),
                confirmation == null ? null : confirmation.payload()));
      }
    }
    return pending;
  }

  /**
   * Returns the content that answers the given confirmation requests. ADK matches each answer to a
   * pending request by id and ignores answers that match none, so only ids returned by {@link
   * #pendingConfirmations(List)} resume a tool.
   */
  public static Content confirmationResponse(ConfirmationDecision... decisions) {
    List<Part> parts = new ArrayList<>();
    for (ConfirmationDecision decision : decisions) {
      Map<String, Object> response = new LinkedHashMap<>();
      response.put("confirmed", decision.confirmed());
      if (decision.payload() != null) {
        response.put("payload", decision.payload());
      }
      parts.add(
          Part.builder()
              .functionResponse(
                  FunctionResponse.builder()
                      .id(decision.functionCallId())
                      .name(Functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME)
                      .response(response)
                      .build())
              .build());
    }
    if (parts.isEmpty()) {
      throw new IllegalArgumentException("at least one decision is required");
    }
    return Content.builder().role("user").parts(parts).build();
  }

  /**
   * Returns the long-running tool calls among the events, other than confirmation requests, in the
   * order ADK emitted them. A call stays open until a later turn delivers its result with {@link
   * #functionResponse(FunctionCall, Map)}; an interim result the tool itself returned does not
   * close it, so the workflow decides from that result whether to wait.
   */
  public static List<FunctionCall> longRunningCalls(List<Event> events) {
    List<FunctionCall> pending = new ArrayList<>();
    for (Event event : events) {
      Set<String> longRunning = event.longRunningToolIds().orElse(Set.of());
      for (FunctionCall call : event.functionCalls()) {
        if (call.id().isPresent()
            && longRunning.contains(call.id().get())
            && !Functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME.equals(
                call.name().orElse(null))) {
          pending.add(call);
        }
      }
    }
    return pending;
  }

  /** Returns the user content that delivers the result of a long-running tool call. */
  public static Content functionResponse(FunctionCall call, Map<String, Object> response) {
    Objects.requireNonNull(call, "call");
    return Content.builder()
        .role("user")
        .parts(
            List.of(
                Part.builder()
                    .functionResponse(
                        FunctionResponse.builder()
                            .id(call.id().orElseThrow())
                            .name(call.name().orElseThrow())
                            .response(response == null ? Map.of() : response)
                            .build())
                    .build()))
        .build();
  }
}
