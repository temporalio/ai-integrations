package io.temporal.langchain4j;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Function;

/** A worker-side chat model that answers each call with the next scripted step. */
final class ScriptedChatModel implements ChatModel {

  private final List<Function<ChatRequest, ChatResponse>> steps;
  final List<ChatRequest> requests = new CopyOnWriteArrayList<>();

  @SafeVarargs
  ScriptedChatModel(Function<ChatRequest, ChatResponse>... steps) {
    this.steps = List.of(steps);
  }

  @Override
  public ChatResponse doChat(ChatRequest request) {
    requests.add(request);
    int index = Math.min(requests.size(), steps.size()) - 1;
    return steps.get(index).apply(request);
  }
}
