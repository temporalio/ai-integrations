package io.temporal.langchain4j.activity;

import io.temporal.activity.ActivityInterface;
import io.temporal.activity.ActivityMethod;

/**
 * Activity that runs one LangChain4j chat request against a {@code StreamingChatModel} registered
 * on the worker and collects its partial output. {@code TemporalStreamingChatModel} schedules it;
 * workflows don't call it directly.
 */
@ActivityInterface
public interface StreamingChatModelActivity {

  /** The activity type name. */
  String STREAMING_CHAT = "langchain4j.StreamingChat";

  /** Streams the request from the model registered under the request's model name. */
  @ActivityMethod(name = STREAMING_CHAT)
  StreamingChatOutput streamingChat(StreamingChatInput input);
}
