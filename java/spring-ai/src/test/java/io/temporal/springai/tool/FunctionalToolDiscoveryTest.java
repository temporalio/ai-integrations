package io.temporal.springai.tool;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.nexusrpc.Service;
import io.temporal.activity.ActivityInterface;
import java.lang.reflect.Proxy;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.annotation.Tool;

class FunctionalToolDiscoveryTest {

  public interface StringFunction extends Function<String, String> {}

  public abstract static class StringSupplier implements Supplier<String> {}

  public interface StringConsumer extends Consumer<String> {}

  @ActivityInterface
  @Service
  public interface Tools {
    @Tool(description = "Return an object")
    Object objectTool();

    @Tool(description = "Return a string")
    String stringTool();

    @Tool(description = "Return a number")
    int numberTool();

    @Tool(description = "Return a function")
    Function<String, String> functionTool();

    @Tool(description = "Return a supplier")
    Supplier<String> supplierTool();

    @Tool(description = "Return a consumer")
    Consumer<String> consumerTool();

    @Tool(description = "Return a function subtype")
    StringFunction functionSubtypeTool();

    @Tool(description = "Return a supplier subtype")
    StringSupplier supplierSubtypeTool();

    @Tool(description = "Return a consumer subtype")
    StringConsumer consumerSubtypeTool();
  }

  @ParameterizedTest
  @ValueSource(booleans = {false, true})
  void discoversOrdinaryToolsAndExcludesFunctionalReturnTypes(boolean nexus) {
    Object stub =
        Proxy.newProxyInstance(
            Tools.class.getClassLoader(),
            new Class<?>[] {Tools.class},
            (proxy, method, args) -> {
              if (method.getName().equals("objectTool")) {
                return Map.of("answer", 42);
              }
              throw new AssertionError("Unexpected invocation: " + method);
            });

    ToolCallback[] callbacks =
        nexus ? NexusToolUtil.fromNexusServiceStub(stub) : ActivityToolUtil.fromActivityStub(stub);

    assertEquals(
        List.of("numberTool", "objectTool", "stringTool"),
        Arrays.stream(callbacks)
            .map(callback -> callback.getToolDefinition().name())
            .sorted()
            .toList());
    ToolCallback objectTool =
        Arrays.stream(callbacks)
            .filter(callback -> callback.getToolDefinition().name().equals("objectTool"))
            .findFirst()
            .orElseThrow();
    assertEquals("{\"answer\":42}", objectTool.call("{}"));
  }
}
