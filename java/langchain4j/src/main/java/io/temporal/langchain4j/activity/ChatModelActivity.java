package io.temporal.langchain4j.activity;

import io.temporal.activity.ActivityInterface;
import io.temporal.activity.ActivityMethod;

/**
 * Activity that runs one LangChain4j chat request against a {@code ChatModel} registered on the
 * worker. {@code TemporalChatModel} schedules it; workflows don't call it directly.
 */
@ActivityInterface
public interface ChatModelActivity {

  /** The activity type name. */
  String CHAT = "langchain4j.Chat";

  /** Sends the request to the chat model registered under {@link ChatInput#model()}. */
  @ActivityMethod(name = CHAT)
  ChatOutput chat(ChatInput input);
}
