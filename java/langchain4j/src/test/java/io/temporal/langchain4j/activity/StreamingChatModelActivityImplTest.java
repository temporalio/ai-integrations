package io.temporal.langchain4j.activity;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialResponseContext;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.chat.response.StreamingHandle;
import io.temporal.activity.ActivityOptions;
import io.temporal.common.RetryOptions;
import io.temporal.failure.ActivityFailure;
import io.temporal.failure.ApplicationFailure;
import io.temporal.langchain4j.internal.ChatCodec;
import io.temporal.testing.TestActivityEnvironment;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/** Worker-side cancellation and validation of streaming model calls. */
@Timeout(30)
class StreamingChatModelActivityImplTest {

  private static final ActivityOptions OPTIONS =
      ActivityOptions.newBuilder()
          .setStartToCloseTimeout(Duration.ofSeconds(10))
          .setHeartbeatTimeout(Duration.ofMillis(300))
          .setRetryOptions(RetryOptions.newBuilder().setMaximumAttempts(1).build())
          .build();
  private static final ChatInput REQUEST =
      ChatCodec.encode("model", ChatRequest.builder().messages(UserMessage.from("Hi")).build());

  private TestActivityEnvironment environment;

  @BeforeEach
  void setUp() {
    environment = TestActivityEnvironment.newInstance();
  }

  @AfterEach
  void tearDown() {
    environment.close();
  }

  private StreamingChatModelActivity register(StreamingChatModel model) {
    environment.registerActivitiesImplementations(
        new StreamingChatModelActivityImpl(Map.of("model", () -> model)));
    return environment.newActivityStub(StreamingChatModelActivity.class, OPTIONS);
  }

  @Test
  void aCancellationCancelsTheProvidersStreamAndReleasesTheActivity() {
    AtomicBoolean cancelled = new AtomicBoolean();
    StreamingHandle handle =
        new StreamingHandle() {
          @Override
          public void cancel() {
            cancelled.set(true);
          }

          @Override
          public boolean isCancelled() {
            return cancelled.get();
          }
        };
    StreamingChatModel neverCompletes =
        new StreamingChatModel() {
          @Override
          public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
            handler.onPartialResponse(
                new PartialResponse("Hel"), new PartialResponseContext(handle));
          }
        };
    StreamingChatModelActivity activity = register(neverCompletes);
    environment.requestCancelActivity();

    assertThrows(
        RuntimeException.class,
        () -> activity.streamingChat(new StreamingChatInput(REQUEST, null)));
    assertTrue(cancelled.get(), "the provider's stream is cancelled");
  }

  @Test
  void aBlankTopicIsNotRetried() {
    StreamingChatModelActivity activity =
        register(
            new StreamingChatModel() {
              @Override
              public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
                throw new AssertionError("must not be called");
              }
            });
    ActivityFailure failure =
        assertThrows(
            ActivityFailure.class,
            () -> activity.streamingChat(new StreamingChatInput(REQUEST, " ")));
    ApplicationFailure cause = (ApplicationFailure) failure.getCause();
    assertEquals(StreamingChatModelActivityImpl.STREAM_ERROR, cause.getType());
    assertTrue(cause.isNonRetryable());
  }
}
