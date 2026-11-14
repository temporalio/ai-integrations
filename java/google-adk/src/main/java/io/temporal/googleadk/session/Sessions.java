package io.temporal.googleadk.session;

import com.google.adk.events.Event;
import com.google.adk.events.EventActions;
import com.google.adk.sessions.BaseSessionService;
import com.google.adk.sessions.Session;
import com.google.adk.sessions.State;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Moves an in-workflow session across continue-as-new.
 *
 * <p>Temporal history is the durable store while a workflow runs, but a new run starts with an
 * empty session service. Export the session before continuing as new, pass the snapshot as the new
 * run's input, and import it into the new run's session service:
 *
 * <pre>{@code
 * Session snapshot = Sessions.exportSession(runner.sessionService(), "app", userId, sessionId);
 * Workflow.continueAsNew(snapshot);
 * // next run
 * Sessions.importSession(runner.sessionService(), snapshot);
 * }</pre>
 *
 * <p>Both helpers block on the session service's results, which is safe inside a workflow only for
 * the in-memory service {@code TemporalAdk.runnerBuilder()} installs, whose calls complete
 * synchronously.
 *
 * <p>The snapshot carries the session's own state and its events, including pending long-running
 * calls and confirmation requests. App-scoped ({@code app:}), user-scoped ({@code user:}) and
 * temporary ({@code temp:}) state is not carried over, because the in-memory service keeps it
 * outside the session. Removed-key markers in event deltas are dropped because they do not survive
 * JSON; the imported state is reconciled with the snapshot after the events are replayed.
 */
public final class Sessions {

  private Sessions() {}

  /** Exports the named session as a snapshot that serializes with the plugin's data converter. */
  public static Session exportSession(
      BaseSessionService sessionService, String appName, String userId, String sessionId) {
    Session session =
        sessionService.getSession(appName, userId, sessionId, Optional.empty()).blockingGet();
    if (session == null) {
      throw new IllegalArgumentException(
          "Session " + sessionId + " for user " + userId + " in app " + appName + " not found");
    }
    return exportSession(session);
  }

  /** Returns a copy of the session with scoped state and removed-key markers stripped. */
  public static Session exportSession(Session session) {
    Objects.requireNonNull(session, "session");
    List<Event> events = new ArrayList<>();
    for (Event event : session.immutableEvents()) {
      events.add(withPortableDelta(event));
    }
    return Session.builder(session.id())
        .appName(session.appName())
        .userId(session.userId())
        .state(portable(session.state()))
        .events(events)
        .lastUpdateTime(session.lastUpdateTime())
        .build();
  }

  /**
   * Recreates the snapshot in the given session service and returns the stored session. The
   * snapshot's id, state, and events are kept; the service must not already hold the session.
   */
  public static Session importSession(BaseSessionService sessionService, Session snapshot) {
    Objects.requireNonNull(sessionService, "sessionService");
    Objects.requireNonNull(snapshot, "snapshot");
    Map<String, Object> state = portable(snapshot.state());
    Session session =
        sessionService
            .createSession(snapshot.appName(), snapshot.userId(), state, snapshot.id())
            .blockingGet();
    for (Event event : snapshot.immutableEvents()) {
      sessionService.appendEvent(session, withPortableDelta(event)).blockingGet();
    }
    // Appending replays each event's state delta; the snapshot's state is the truth, so restore
    // keys the replay removed and drop keys it resurrected.
    session
        .state()
        .keySet()
        .removeIf(
            key ->
                !state.containsKey(key)
                    && !key.startsWith(State.APP_PREFIX)
                    && !key.startsWith(State.USER_PREFIX));
    session.state().putAll(state);
    return session;
  }

  private static Event withPortableDelta(Event event) {
    EventActions actions = event.actions();
    if (actions == null || actions.stateDelta() == null || actions.stateDelta().isEmpty()) {
      return event;
    }
    Map<String, Object> delta = portable(actions.stateDelta());
    if (delta.size() == actions.stateDelta().size()) {
      return event;
    }
    return event.toBuilder().actions(actions.toBuilder().stateDelta(delta).build()).build();
  }

  private static Map<String, Object> portable(Map<String, Object> state) {
    Map<String, Object> copy = new LinkedHashMap<>();
    if (state == null) {
      return copy;
    }
    state.forEach(
        (key, value) -> {
          if (key.startsWith(State.APP_PREFIX)
              || key.startsWith(State.USER_PREFIX)
              || key.startsWith(State.TEMP_PREFIX)
              || value == State.REMOVED) {
            return;
          }
          copy.put(key, value);
        });
    return copy;
  }
}
