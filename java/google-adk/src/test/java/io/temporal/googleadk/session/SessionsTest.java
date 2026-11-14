package io.temporal.googleadk.session;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.adk.events.Event;
import com.google.adk.events.EventActions;
import com.google.adk.sessions.InMemorySessionService;
import com.google.adk.sessions.Session;
import com.google.adk.sessions.SessionException;
import com.google.adk.sessions.State;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.temporal.api.common.v1.Payload;
import io.temporal.common.converter.DataConverter;
import io.temporal.googleadk.converter.GoogleAdkDataConverter;
import java.time.Instant;
import java.time.InstantSource;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/** A session survives export, serialization, and import with its state reconciled. */
class SessionsTest {

  private static InMemorySessionService newService() {
    AtomicInteger ids = new AtomicInteger();
    return new InMemorySessionService(
        InMemorySessionService.DuplicateSessionIdBehavior.REJECT,
        InstantSource.fixed(Instant.ofEpochSecond(100)),
        () -> "uuid-" + ids.getAndIncrement());
  }

  private static Event event(String id, long timestamp, Map<String, Object> delta) {
    return Event.builder()
        .id(id)
        .invocationId("inv-1")
        .author("assistant")
        .timestamp(timestamp)
        .content(Content.fromParts(Part.fromText("step " + id)))
        .longRunningToolIds(Set.of("fc-" + id))
        .actions(EventActions.builder().stateDelta(delta).build())
        .build();
  }

  @Test
  void exportImport_keepsEventsAndReconcilesState() {
    InMemorySessionService source = newService();
    Session created =
        source
            .createSession("app", "user-1", Map.of("x", 1, "user:tier", "gold"), "s1")
            .blockingGet();
    Map<String, Object> delta1 = new HashMap<>();
    delta1.put("k", "v1");
    delta1.put("gone", "g");
    delta1.put("app:theme", "dark");
    source.appendEvent(created, event("e1", 1_000L, delta1)).blockingGet();
    Map<String, Object> delta2 = new HashMap<>();
    delta2.put("gone", State.REMOVED);
    delta2.put("temp:scratch", 1);
    source.appendEvent(created, event("e2", 2_000L, delta2)).blockingGet();

    Session snapshot = Sessions.exportSession(source, "app", "user-1", "s1");

    assertEquals(
        Map.of("x", 1, "k", "v1"), snapshot.state(), "scoped and removed keys are dropped");
    assertEquals(2, snapshot.immutableEvents().size());
    assertFalse(
        snapshot.immutableEvents().get(1).actions().stateDelta().containsKey("gone"),
        "removed-key markers do not survive JSON and are dropped");

    DataConverter converter = GoogleAdkDataConverter.newDefaultInstance();
    Payload payload = converter.toPayload(snapshot).orElseThrow();
    Session restored = converter.fromPayload(payload, Session.class, Session.class);

    InMemorySessionService target = newService();
    Session imported = Sessions.importSession(target, restored);

    assertEquals(Map.of("x", 1, "k", "v1"), imported.state(), "replayed deltas are reconciled");
    Session stored = target.getSession("app", "user-1", "s1", Optional.empty()).blockingGet();
    assertEquals(Map.of("x", 1, "k", "v1"), stored.state());
    assertEquals(List.of("e1", "e2"), stored.immutableEvents().stream().map(Event::id).toList());
    assertEquals(
        Set.of("fc-e2"), stored.immutableEvents().get(1).longRunningToolIds().orElseThrow());
    assertEquals(Instant.ofEpochMilli(2_000L), stored.lastUpdateTime());
  }

  @Test
  void importSession_rejectsAnExistingSession() {
    InMemorySessionService service = newService();
    Session snapshot =
        Sessions.exportSession(
            service.createSession("app", "user-1", Map.of(), "s1").blockingGet());

    SessionException failure =
        assertThrows(SessionException.class, () -> Sessions.importSession(service, snapshot));

    assertTrue(failure.getMessage().contains("already exists"), failure.getMessage());
  }

  @Test
  void exportSession_failsForAnUnknownSession() {
    IllegalArgumentException failure =
        assertThrows(
            IllegalArgumentException.class,
            () -> Sessions.exportSession(newService(), "app", "user-1", "missing"));

    assertTrue(failure.getMessage().contains("missing"));
  }
}
