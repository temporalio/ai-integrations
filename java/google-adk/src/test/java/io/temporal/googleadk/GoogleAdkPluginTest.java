package io.temporal.googleadk;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

import com.google.adk.models.LlmRequest;
import com.google.adk.tools.BaseTool;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.temporal.api.common.v1.Payload;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.common.converter.CodecDataConverter;
import io.temporal.common.converter.DataConverter;
import io.temporal.common.converter.DataConverterException;
import io.temporal.common.converter.DefaultDataConverter;
import io.temporal.common.converter.EncodingKeys;
import io.temporal.common.converter.GlobalDataConverter;
import io.temporal.common.converter.JacksonJsonPayloadConverter;
import io.temporal.common.converter.NullPayloadConverter;
import io.temporal.googleadk.activity.McpActivitiesImpl;
import io.temporal.googleadk.activity.ModelActivityImpl;
import io.temporal.googleadk.converter.GoogleAdkDataConverter;
import io.temporal.worker.Worker;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

/** Plugin wiring: activity registration and data converter customization. */
class GoogleAdkPluginTest {

  private static final LlmRequest REQUEST =
      LlmRequest.builder()
          .model("gemini-2.5-flash")
          .contents(List.of(Content.fromParts(Part.fromText("hi"))))
          .build();

  /** A request carrying a live tool, as every request of an agent with tools does. */
  private static final LlmRequest REQUEST_WITH_TOOL =
      REQUEST.toBuilder().tools(Map.of("echo", new BaseTool("echo", "Echoes.") {})).build();

  @Test
  void name_isStable() {
    assertEquals("io.temporal.google-adk", GoogleAdkPlugin.NAME);
    assertEquals(GoogleAdkPlugin.NAME, GoogleAdkPlugin.newDefaultInstance().getName());
  }

  @Test
  void initializeWorker_registersTheModelAndMcpActivities() {
    Worker worker = mock(Worker.class);

    GoogleAdkPlugin.newDefaultInstance().initializeWorker("queue", worker);

    ArgumentCaptor<Object> captor = ArgumentCaptor.forClass(Object.class);
    verify(worker).registerActivitiesImplementations(captor.capture(), captor.capture());
    List<Object> registered = captor.getAllValues();
    assertEquals(2, registered.size());
    assertInstanceOf(ModelActivityImpl.class, registered.get(0));
    assertInstanceOf(McpActivitiesImpl.class, registered.get(1));
  }

  @Test
  void configureWorkflowClient_installsTheAdkPayloadConverter() {
    WorkflowClientOptions.Builder builder = WorkflowClientOptions.newBuilder();

    GoogleAdkPlugin.newDefaultInstance().configureWorkflowClient(builder);

    DataConverter converter = builder.validateAndBuildWithDefaults().getDataConverter();
    Payload payload = converter.toPayload(REQUEST).orElseThrow();
    assertEquals(REQUEST, converter.fromPayload(payload, LlmRequest.class, LlmRequest.class));
  }

  @Test
  void configureWorkflowClient_withoutAConfiguredConverter_leavesTheSharedConvertersAlone() {
    WorkflowClientOptions.Builder builder = WorkflowClientOptions.newBuilder();
    assertNull(builder.build().getDataConverter(), "nothing configured yet");

    GoogleAdkPlugin.newDefaultInstance().configureWorkflowClient(builder);

    DataConverter converter = builder.build().getDataConverter();
    assertTrue(roundTripsAToolCarryingRequest(converter), "the client understands ADK types");
    assertNotSame(DefaultDataConverter.STANDARD_INSTANCE, converter);
    assertFalse(
        roundTripsAToolCarryingRequest(GlobalDataConverter.get()),
        "the global converter is not modified");
    assertFalse(
        roundTripsAToolCarryingRequest(DefaultDataConverter.STANDARD_INSTANCE),
        "the SDK's own converter is not modified");
  }

  @Test
  void withAdkTypes_neverModifiesTheSharedConverters() {
    DataConverter fromStandard =
        GoogleAdkDataConverter.withAdkTypes(DefaultDataConverter.STANDARD_INSTANCE);
    DataConverter fromGlobal = GoogleAdkDataConverter.withAdkTypes(GlobalDataConverter.get());

    assertNotSame(DefaultDataConverter.STANDARD_INSTANCE, fromStandard);
    assertNotSame(GlobalDataConverter.get(), fromGlobal);
    assertTrue(roundTripsAToolCarryingRequest(fromStandard));
    assertTrue(roundTripsAToolCarryingRequest(fromGlobal));
    assertFalse(roundTripsAToolCarryingRequest(DefaultDataConverter.STANDARD_INSTANCE));
  }

  @Test
  void withAdkTypes_extendsAGloballyRegisteredApplicationConverterInPlace() {
    DefaultDataConverter own =
        new DefaultDataConverter(new NullPayloadConverter(), new JacksonJsonPayloadConverter());
    GlobalDataConverter.register(own);
    try {
      DataConverter forClientWithoutConverter =
          GoogleAdkDataConverter.withAdkTypes((DataConverter) null);
      DataConverter forClientWithTheSameInstance =
          GoogleAdkDataConverter.withAdkTypes((DataConverter) own);

      assertSame(
          own, forClientWithoutConverter, "the application's converter is extended, not replaced");
      assertSame(own, forClientWithTheSameInstance);
      assertTrue(roundTripsAToolCarryingRequest(own));
      assertEquals(
          "json/plain",
          encodingOf(own.toPayload(new byte[] {1}).orElseThrow()),
          "the application's own payload converters are kept");
    } finally {
      GlobalDataConverter.register(DefaultDataConverter.STANDARD_INSTANCE);
    }
    assertFalse(roundTripsAToolCarryingRequest(DefaultDataConverter.STANDARD_INSTANCE));
  }

  @Test
  void withAdkTypes_extendsAnApplicationConverterInPlace() {
    DefaultDataConverter own =
        new DefaultDataConverter(new NullPayloadConverter(), new JacksonJsonPayloadConverter());
    assertFalse(roundTripsAToolCarryingRequest(own));

    DefaultDataConverter extended = GoogleAdkDataConverter.withAdkTypes(own);

    assertSame(own, extended, "the SDK offers no copy, so the converter is extended in place");
    assertTrue(roundTripsAToolCarryingRequest(extended));
    assertEquals(
        "json/plain",
        encodingOf(extended.toPayload(new byte[] {1}).orElseThrow()),
        "the application's own list of payload converters is kept");
  }

  @Test
  void customizeDataConverterFalse_leavesTheClientConverterAlone() {
    WorkflowClientOptions.Builder builder = WorkflowClientOptions.newBuilder();

    GoogleAdkPlugin.newBuilder()
        .customizeDataConverter(false)
        .build()
        .configureWorkflowClient(builder);

    assertNull(builder.build().getDataConverter(), "the plugin must not touch the converter");
  }

  @Test
  void withAdkTypes_rejectsConvertersItCannotExtend() {
    DataConverter codec =
        new CodecDataConverter(DefaultDataConverter.newDefaultInstance(), List.of());

    IllegalStateException failure =
        assertThrows(IllegalStateException.class, () -> GoogleAdkDataConverter.withAdkTypes(codec));

    assertTrue(failure.getMessage().contains("GoogleAdkDataConverter.newDefaultInstance()"));
  }

  @Test
  void configureWorkflowClient_failsLoudlyOnAConverterItCannotExtend() {
    WorkflowClientOptions.Builder builder =
        WorkflowClientOptions.newBuilder()
            .setDataConverter(
                new CodecDataConverter(DefaultDataConverter.newDefaultInstance(), List.of()));

    assertThrows(
        IllegalStateException.class,
        () -> GoogleAdkPlugin.newDefaultInstance().configureWorkflowClient(builder));
  }

  /**
   * Whether the converter writes a request that carries a tool and reads it back. Temporal's own
   * JSON converter serializes the request's tool instances, which it cannot read back.
   */
  private static boolean roundTripsAToolCarryingRequest(DataConverter converter) {
    try {
      Payload payload = converter.toPayload(REQUEST_WITH_TOOL).orElseThrow();
      converter.fromPayload(payload, LlmRequest.class, LlmRequest.class);
      return true;
    } catch (DataConverterException e) {
      return false;
    }
  }

  private static String encodingOf(Payload payload) {
    return payload.getMetadataOrThrow(EncodingKeys.METADATA_ENCODING_KEY).toStringUtf8();
  }
}
