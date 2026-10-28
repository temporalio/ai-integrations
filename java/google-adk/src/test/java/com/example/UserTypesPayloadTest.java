package com.example;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.adk.events.Event;
import com.google.adk.events.EventActions;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.adk.sessions.Session;
import com.google.adk.tools.Annotations.Schema;
import com.google.adk.tools.FunctionTool;
import com.google.genai.types.Content;
import com.google.genai.types.FunctionCall;
import com.google.genai.types.Part;
import io.temporal.api.common.v1.Payload;
import io.temporal.common.converter.DataConverter;
import io.temporal.common.converter.DefaultDataConverter;
import io.temporal.googleadk.activity.CallMcpToolInput;
import io.temporal.googleadk.converter.GoogleAdkDataConverter;
import io.temporal.googleadk.converter.GoogleAdkPayloadConverter;
import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

/**
 * Application types outside the plugin's package, with and without ADK members, go through the
 * plugin's data converter the way they go through Temporal's default one.
 */
class UserTypesPayloadTest {

  private final DataConverter converter = GoogleAdkDataConverter.newDefaultInstance();

  /** A workflow input that carries a session snapshot next to plain values. */
  record ConversationInput(
      String label, Session session, List<Event> events, Optional<LlmResponse> last) {}

  /** A plain class with private fields and no accessors, as Temporal's default allows. */
  static final class FieldPojo {
    private String name;
    private int count;
    private Instant when;
  }

  public static final class Tools {
    private Tools() {}

    public static Map<String, Object> add(@Schema(name = "a") int a, @Schema(name = "b") int b) {
      return Map.of("sum", a + b);
    }
  }

  private static Event event() {
    return Event.builder()
        .id("e1")
        .invocationId("inv-1")
        .author("assistant")
        .timestamp(1234L)
        .content(
            Content.builder()
                .role("model")
                .parts(
                    List.of(
                        Part.builder()
                            .functionCall(
                                FunctionCall.builder()
                                    .id("fc1")
                                    .name("add")
                                    .args(Map.of("a", 1))
                                    .build())
                            .build()))
                .build())
        .longRunningToolIds(Set.of("fc1"))
        .actions(EventActions.builder().stateDelta(Map.of("k", "v")).build())
        .build();
  }

  private static Session session() {
    return Session.builder("s1")
        .appName("app")
        .userId("user-1")
        .state(Map.of("x", 1))
        .events(List.of(event()))
        .lastUpdateTime(Instant.ofEpochSecond(5))
        .build();
  }

  private <T> T roundTrip(T value, Class<T> type) {
    Payload payload = converter.toPayload(value).orElseThrow();
    return converter.fromPayload(payload, type, type);
  }

  @Test
  void recordWithAdkMembers_roundTrips() {
    LlmResponse last =
        LlmResponse.builder()
            .content(Content.fromParts(Part.fromText("bye")))
            .turnComplete(true)
            .build();
    ConversationInput input =
        new ConversationInput("L", session(), List.of(event()), Optional.of(last));

    ConversationInput back = roundTrip(input, ConversationInput.class);

    assertEquals("L", back.label());
    assertEquals(session().toJson(), back.session().toJson());
    assertEquals(event().toJson(), back.events().get(0).toJson());
    assertEquals(Optional.of(last), back.last());
  }

  @Test
  void eventArray_roundTrips() {
    Event[] back = roundTrip(new Event[] {event(), event()}, Event[].class);

    assertEquals(2, back.length);
    assertEquals(event().toJson(), back[1].toJson());
  }

  @Test
  void fieldOnlyPojo_keepsTemporalsFieldVisibilityAndDateFormat() {
    FieldPojo pojo = new FieldPojo();
    pojo.name = "n";
    pojo.count = 3;
    pojo.when = Instant.ofEpochSecond(1);

    Payload payload = converter.toPayload(pojo).orElseThrow();
    String json = payload.getData().toStringUtf8();
    FieldPojo back = converter.fromPayload(payload, FieldPojo.class, FieldPojo.class);

    assertTrue(json.contains("\"when\":\"1970-01-01T00:00:01Z\""), json);
    assertEquals("n", back.name);
    assertEquals(3, back.count);
    assertEquals(Instant.ofEpochSecond(1), back.when);
  }

  /** A class whose optional fields are only ever set by Jackson. */
  static final class OptionalFieldPojo {
    private Optional<String> maybe;
    private String plain;
  }

  @Test
  void userTypes_keepTemporalsDefaultWireFormat() {
    OptionalFieldPojo pojo = new OptionalFieldPojo();
    pojo.maybe = Optional.empty();
    pojo.plain = null;
    DataConverter temporalDefault = DefaultDataConverter.newDefaultInstance();

    Payload payload = converter.toPayload(pojo).orElseThrow();
    OptionalFieldPojo back =
        converter.fromPayload(payload, OptionalFieldPojo.class, OptionalFieldPojo.class);

    assertEquals(
        temporalDefault.toPayload(pojo).orElseThrow().getData().toStringUtf8(),
        payload.getData().toStringUtf8(),
        "types outside ADK serialize exactly as with Temporal's default converter");
    assertEquals(Optional.empty(), back.maybe, "an absent optional reads back as empty, not null");
    assertEquals(null, back.plain);
  }

  @Test
  void nullToolArgumentValues_roundTrip() {
    Map<String, Object> args = new HashMap<>();
    args.put("city", null);
    args.put("n", 2);

    CallMcpToolInput back =
        roundTrip(new CallMcpToolInput("docs", "lookup", args), CallMcpToolInput.class);

    assertTrue(back.args().containsKey("city"));
    assertEquals(null, back.args().get("city"));
    assertEquals(2, back.args().get("n"));
  }

  @Test
  void requestWithToolInstances_omitsThemAndRoundTrips() throws Exception {
    LlmRequest.Builder builder =
        LlmRequest.builder()
            .model("gemini-2.5-flash")
            .contents(List.of(Content.fromParts(Part.fromText("add 1 and 2"))));
    builder.appendTools(List.of(FunctionTool.create(Tools.class, "add")));
    LlmRequest request = builder.build();

    Payload payload = converter.toPayload(request).orElseThrow();
    String json = payload.getData().toStringUtf8();
    LlmRequest back = converter.fromPayload(payload, LlmRequest.class, LlmRequest.class);

    assertFalse(json.contains("\"tools\":{"), json);
    assertFalse(json.contains("null"), json);
    assertEquals(request.toJson(), back.toJson());
    assertTrue(back.tools().isEmpty(), "tool instances are rebuilt on the worker, not shipped");
    assertEquals(
        json,
        GoogleAdkPayloadConverter.newObjectMapper().writeValueAsString(request),
        "the mapper the converter exposes writes the same JSON");
  }
}
