package io.temporal.langchain4j.activity;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;
import io.temporal.activity.Activity;
import io.temporal.failure.ApplicationFailure;
import io.temporal.langchain4j.internal.ChatCodec;
import io.temporal.langchain4j.internal.Failures;
import io.temporal.langchain4j.internal.Heartbeats;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/** Runs chat requests against the chat models registered with {@code LangChain4jPlugin}. */
public final class ChatModelActivityImpl implements ChatModelActivity {

  /** The failure type when a workflow names a model that the worker doesn't serve. */
  public static final String UNKNOWN_MODEL = "LangChain4jUnknownModel";

  private final Map<String, Supplier<ChatModel>> registrations;
  private final Map<String, ChatModel> models = new ConcurrentHashMap<>();

  /** Creates the activity for the given registrations; each model is created on first use. */
  public ChatModelActivityImpl(Map<String, Supplier<ChatModel>> registrations) {
    this.registrations = Map.copyOf(Objects.requireNonNull(registrations, "registrations"));
  }

  @Override
  public ChatOutput chat(ChatInput input) {
    ChatModel model = model(input.model());
    ChatResponse response;
    try (Heartbeats ignored = Heartbeats.start(Activity.getExecutionContext())) {
      response = model.chat(ChatCodec.decode(input));
    } catch (RuntimeException e) {
      throw Failures.modelFailure(input.model(), e);
    }
    return ChatCodec.encode(response);
  }

  private ChatModel model(String name) {
    Supplier<ChatModel> registration = registrations.get(name);
    if (registration == null) {
      throw ApplicationFailure.newNonRetryableFailure(
          "No chat model is registered under '"
              + name
              + "'. Register it with LangChain4jPlugin.newBuilder().chatModel(...); registered: "
              + registrations.keySet(),
          UNKNOWN_MODEL);
    }
    return models.computeIfAbsent(name, key -> Objects.requireNonNull(registration.get(), key));
  }
}
