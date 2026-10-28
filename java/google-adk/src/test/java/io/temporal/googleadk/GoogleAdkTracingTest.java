package io.temporal.googleadk;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanContext;
import io.opentelemetry.api.trace.TraceFlags;
import io.opentelemetry.api.trace.TraceState;
import io.opentelemetry.context.Context;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import java.util.Map;
import org.junit.jupiter.api.Test;

class GoogleAdkTracingTest {
  @Test
  void traceContextPreservesRemoteParentAndTraceStateThroughSdkThreadHandoffs() {
    try (GoogleAdkTracing tracing = GoogleAdkTracing.create(InMemorySpanExporter.create())) {
      SpanContext original =
          SpanContext.createFromRemoteParent(
              "11111111111111111111111111111111",
              "2222222222222222",
              TraceFlags.getSampled(),
              TraceState.builder().put("vendor", "example").build());
      NativeTracingPropagation propagation = tracing.propagation;
      Context restored =
          (Context)
              propagation.deserializeContext(
                  propagation.serializeContext(Context.root().with(Span.wrap(original))));
      assertEquals(original, Span.fromContext(restored).getSpanContext());
      propagation.setCurrentContext(restored);
      Context handedOff = (Context) propagation.getCurrentContext();
      assertEquals(original, Span.fromContext(handedOff).getSpanContext());
      propagation.setCurrentContext(Context.root());
    }
  }

  @Test
  void absentOrMalformedTraceHeadersDoNotCreateAParent() {
    try (GoogleAdkTracing tracing = GoogleAdkTracing.create(InMemorySpanExporter.create())) {
      NativeTracingPropagation propagation = tracing.propagation;
      for (Map<String, io.temporal.api.common.v1.Payload> headers :
          java.util.List.of(
              Map.<String, io.temporal.api.common.v1.Payload>of(),
              Map.of(
                  "io.temporal.googleadk.traceparent.traceparent",
                  io.temporal.api.common.v1.Payload.newBuilder()
                      .setData(com.google.protobuf.ByteString.copyFromUtf8("not-a-traceparent"))
                      .build()))) {
        Context restored = (Context) propagation.deserializeContext(headers);
        assertFalse(Span.fromContext(restored).getSpanContext().isValid());
      }
      assertTrue(propagation.serializeContext(Context.root()).isEmpty());
    }
  }

  @Test
  void wrapperDelegatesOtherInstrumentationAndDoesNotOwnApplicationTelemetry() {
    InMemorySpanExporter applicationExporter = InMemorySpanExporter.create();
    try (OpenTelemetrySdk application =
        OpenTelemetrySdk.builder()
            .setTracerProvider(
                SdkTracerProvider.builder()
                    .addSpanProcessor(SimpleSpanProcessor.create(applicationExporter))
                    .build())
            .build()) {
      try (GoogleAdkTracing tracing = GoogleAdkTracing.create(InMemorySpanExporter.create())) {
        OpenTelemetry wrapped = GoogleAdkTracing.wrap(application);
        Span ordinary =
            wrapped
                .tracerBuilder("application")
                .setInstrumentationVersion("test-version")
                .setSchemaUrl("https://example.test/schema")
                .build()
                .spanBuilder("ordinary")
                .startSpan();
        ordinary.end();
        Span nativeOutside =
            wrapped
                .tracerBuilder("gcp.vertex.agent")
                .setInstrumentationVersion("native-version")
                .setSchemaUrl("https://example.test/native-schema")
                .build()
                .spanBuilder("outside")
                .startSpan();
        nativeOutside.end();
        assertEquals(
            "test-version",
            applicationExporter
                .getFinishedSpanItems()
                .get(0)
                .getInstrumentationScopeInfo()
                .getVersion());
        assertEquals(
            "https://example.test/schema",
            applicationExporter
                .getFinishedSpanItems()
                .get(0)
                .getInstrumentationScopeInfo()
                .getSchemaUrl());
        assertEquals(
            "native-version",
            applicationExporter
                .getFinishedSpanItems()
                .get(1)
                .getInstrumentationScopeInfo()
                .getVersion());
        assertEquals(
            "https://example.test/native-schema",
            applicationExporter
                .getFinishedSpanItems()
                .get(1)
                .getInstrumentationScopeInfo()
                .getSchemaUrl());
      }
      application.getTracer("application").spanBuilder("after-close").startSpan().end();
      assertEquals(3, applicationExporter.getFinishedSpanItems().size());
    }
  }
}
