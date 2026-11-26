package io.temporal.langchain4j;

import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.BiConsumer;

/**
 * A worker-side streaming chat model that answers each call with the next scripted step, on a
 * thread of its own as provider clients do.
 */
final class ScriptedStreamingChatModel implements StreamingChatModel {

  private final List<BiConsumer<ChatRequest, StreamingChatResponseHandler>> steps;
  final List<ChatRequest> requests = new CopyOnWriteArrayList<>();

  @SafeVarargs
  ScriptedStreamingChatModel(BiConsumer<ChatRequest, StreamingChatResponseHandler>... steps) {
    this.steps = List.of(steps);
  }

  @Override
  public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
    requests.add(request);
    BiConsumer<ChatRequest, StreamingChatResponseHandler> step =
        steps.get(Math.min(requests.size(), steps.size()) - 1);
    Thread thread = new Thread(() -> step.accept(request, handler), "scripted-stream");
    thread.setDaemon(true);
    thread.start();
  }
}
