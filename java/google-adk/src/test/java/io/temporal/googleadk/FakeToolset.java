package io.temporal.googleadk;

import com.google.adk.agents.ReadonlyContext;
import com.google.adk.tools.BaseTool;
import com.google.adk.tools.BaseToolset;
import com.google.adk.tools.ToolContext;
import com.google.genai.types.FunctionDeclaration;
import com.google.genai.types.Schema;
import com.google.genai.types.Type;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.core.Single;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

/** A worker-side toolset standing in for an MCP server: one {@code echo} tool. */
final class FakeToolset implements BaseToolset {

  static final AtomicInteger created = new AtomicInteger();
  static final AtomicInteger closed = new AtomicInteger();
  static final AtomicInteger calls = new AtomicInteger();

  /** Counted down when the {@code wait} tool starts; the tool returns once {@link #gate} opens. */
  static volatile CountDownLatch waiting = new CountDownLatch(1);

  static volatile CountDownLatch gate = new CountDownLatch(1);

  FakeToolset() {
    created.incrementAndGet();
  }

  static void reset() {
    created.set(0);
    closed.set(0);
    calls.set(0);
    waiting = new CountDownLatch(1);
    gate = new CountDownLatch(1);
  }

  @Override
  public Flowable<BaseTool> getTools(ReadonlyContext readonlyContext) {
    return Flowable.just(new EchoTool(), new WaitTool());
  }

  @Override
  public void close() {
    closed.incrementAndGet();
  }

  /** Blocks until the gate opens, to hold a call in flight. */
  private static final class WaitTool extends BaseTool {
    WaitTool() {
      super("wait", "Waits for the test to release it.");
    }

    @Override
    public Optional<FunctionDeclaration> declaration() {
      return Optional.of(
          FunctionDeclaration.builder()
              .name("wait")
              .description("Waits for the test to release it.")
              .parameters(Schema.builder().type(Type.Known.OBJECT).properties(Map.of()).build())
              .build());
    }

    @Override
    public Single<Map<String, Object>> runAsync(Map<String, Object> args, ToolContext toolContext) {
      return Single.fromCallable(
          () -> {
            waiting.countDown();
            if (!gate.await(30, java.util.concurrent.TimeUnit.SECONDS)) {
              throw new IllegalStateException("gate never opened");
            }
            return Map.of("waited", true);
          });
    }
  }

  private static final class EchoTool extends BaseTool {
    EchoTool() {
      super("echo", "Echoes the given text.");
    }

    @Override
    public Optional<FunctionDeclaration> declaration() {
      return Optional.of(
          FunctionDeclaration.builder()
              .name("echo")
              .description("Echoes the given text.")
              .parameters(
                  Schema.builder()
                      .type(Type.Known.OBJECT)
                      .properties(
                          Map.of(
                              "text",
                              Schema.builder()
                                  .type(Type.Known.STRING)
                                  .description("Text to echo")
                                  .build()))
                      .required(List.of("text"))
                      .build())
              .build());
    }

    @Override
    public Single<Map<String, Object>> runAsync(Map<String, Object> args, ToolContext toolContext) {
      calls.incrementAndGet();
      return Single.just(Map.of("echo", String.valueOf(args.get("text"))));
    }
  }
}
