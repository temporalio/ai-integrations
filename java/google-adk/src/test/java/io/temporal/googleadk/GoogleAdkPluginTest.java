package io.temporal.googleadk;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

import com.google.adk.models.LlmRequest;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.temporal.api.common.v1.Payload;
import io.temporal.client.WorkflowClientOptions;
import io.temporal.common.converter.CodecDataConverter;
import io.temporal.common.converter.DataConverter;
import io.temporal.common.converter.DefaultDataConverter;
import io.temporal.googleadk.activity.McpActivitiesImpl;
import io.temporal.googleadk.activity.ModelActivityImpl;
import io.temporal.googleadk.converter.GoogleAdkDataConverter;
import io.temporal.worker.Worker;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

/** Plugin wiring: activity registration and data converter customization. */
class GoogleAdkPluginTest {

  private static final LlmRequest REQUEST =
      LlmRequest.builder()
          .model("gemini-2.5-flash")
          .contents(List.of(Content.fromParts(Part.fromText("hi"))))
          .build();

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
  void configureWorkflowClient_withoutAConfiguredConverter_extendsTheGlobalDefault() {
    WorkflowClientOptions.Builder builder = WorkflowClientOptions.newBuilder();
    assertEquals(null, builder.build().getDataConverter(), "nothing configured yet");

    GoogleAdkPlugin.newDefaultInstance().configureWorkflowClient(builder);

    DataConverter converter = builder.build().getDataConverter();
    Payload payload = converter.toPayload(REQUEST).orElseThrow();
    assertEquals(REQUEST, converter.fromPayload(payload, LlmRequest.class, LlmRequest.class));
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
}
