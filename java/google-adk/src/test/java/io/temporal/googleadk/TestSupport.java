package io.temporal.googleadk;

import com.google.adk.events.Event;
import com.google.adk.models.LlmRequest;
import com.google.genai.types.Content;
import com.google.genai.types.FunctionResponse;
import io.temporal.api.history.v1.HistoryEvent;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowStub;
import io.temporal.common.WorkflowExecutionHistory;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

/** Small assertions shared by the workflow tests. */
final class TestSupport {

  private TestSupport() {}

  static WorkflowExecutionHistory history(WorkflowClient client, Object workflowStub) {
    return client.fetchHistory(WorkflowStub.fromTyped(workflowStub).getExecution().getWorkflowId());
  }

  /** Returns the activity type names of every scheduled activity, in history order. */
  static List<String> scheduledActivityTypes(WorkflowExecutionHistory history) {
    return history.getEvents().stream()
        .filter(HistoryEvent::hasActivityTaskScheduledEventAttributes)
        .map(e -> e.getActivityTaskScheduledEventAttributes().getActivityType().getName())
        .collect(Collectors.toList());
  }

  /** Returns the activity ids of every scheduled activity, in history order. */
  static List<String> scheduledActivityIds(WorkflowExecutionHistory history) {
    return history.getEvents().stream()
        .filter(HistoryEvent::hasActivityTaskScheduledEventAttributes)
        .map(e -> e.getActivityTaskScheduledEventAttributes().getActivityId())
        .collect(Collectors.toList());
  }

  /** Returns the duration of every started timer, in history order. */
  static List<Duration> timerDurations(WorkflowExecutionHistory history) {
    return history.getEvents().stream()
        .filter(HistoryEvent::hasTimerStartedEventAttributes)
        .map(e -> e.getTimerStartedEventAttributes().getStartToFireTimeout())
        .map(timeout -> Duration.ofSeconds(timeout.getSeconds(), timeout.getNanos()))
        .collect(Collectors.toList());
  }

  /** Returns the text of the last event that has text content. */
  static String lastText(List<Event> events) {
    for (int i = events.size() - 1; i >= 0; i--) {
      String text = events.get(i).content().map(Content::text).orElse(null);
      if (text != null && !text.isEmpty()) {
        return text;
      }
    }
    return "";
  }

  /** Returns whether the request's conversation contains a function response for the tool. */
  static boolean hasFunctionResponse(LlmRequest request, String toolName) {
    return request.contents().stream()
        .flatMap(content -> content.parts().orElse(List.of()).stream())
        .map(part -> part.functionResponse())
        .anyMatch(fr -> fr.map(FunctionResponse::name).flatMap(n -> n).orElse("").equals(toolName));
  }

  /** Returns the latest function response for the tool in the request's conversation, if any. */
  static Optional<Map<String, Object>> functionResponse(LlmRequest request, String toolName) {
    List<Map<String, Object>> responses =
        request.contents().stream()
            .flatMap(content -> content.parts().orElse(List.of()).stream())
            .flatMap(part -> part.functionResponse().stream())
            .filter(fr -> fr.name().orElse("").equals(toolName))
            .map(fr -> fr.response().orElse(Map.of()))
            .collect(Collectors.toList());
    return responses.isEmpty()
        ? Optional.empty()
        : Optional.of(responses.get(responses.size() - 1));
  }

  /** Returns whether the request was built for the named agent. */
  static boolean isAgent(LlmRequest request, String agentName) {
    return request.getSystemInstructions().stream()
        .anyMatch(instruction -> instruction.contains("\"" + agentName + "\""));
  }
}
