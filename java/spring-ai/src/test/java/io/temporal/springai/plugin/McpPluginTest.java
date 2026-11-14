package io.temporal.springai.plugin;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

import io.modelcontextprotocol.client.McpSyncClient;
import io.modelcontextprotocol.spec.McpSchema;
import io.temporal.springai.mcp.McpClientActivityImpl;
import io.temporal.worker.Worker;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.ResolvableType;

class McpPluginTest {

  @Test
  void discoversTypedClientListsAndDeduplicatesIndividualBeanAliases() {
    try (var context = new AnnotationConfigApplicationContext(ClientLists.class)) {
      Object alpha = context.getBean("mcpSyncClients", List.class).get(0);
      context.getBeanFactory().registerSingleton("alphaAlias", alpha);
      McpPlugin plugin = new McpPlugin();
      plugin.setApplicationContext(context);
      Worker worker = mock(Worker.class);

      plugin.initializeWorker("mcp-tq", worker);

      ArgumentCaptor<Object> captor = ArgumentCaptor.forClass(Object.class);
      verify(worker).registerActivitiesImplementations(captor.capture());
      McpClientActivityImpl activity = (McpClientActivityImpl) captor.getValue();
      assertEquals(Set.of("alpha", "beta", "gamma"), activity.getClientInfo().keySet());
    }
  }

  @Test
  void distinctClientsWithTheSameNameAcrossListsAndIndividualBeansAreRejected() {
    try (var context = new AnnotationConfigApplicationContext(ClientLists.class)) {
      context.getBeanFactory().registerSingleton("duplicateBeta", mockClientNamed("beta"));
      McpPlugin plugin = new McpPlugin();
      plugin.setApplicationContext(context);

      IllegalArgumentException thrown =
          assertThrows(
              IllegalArgumentException.class,
              () -> plugin.initializeWorker("mcp-tq", mock(Worker.class)));
      assertTrue(thrown.getMessage().contains("beta"));
    }
  }

  @Test
  void clientListsAppearLate_registeredViaAfterSingletonsInstantiated() {
    ApplicationContext context = mock(ApplicationContext.class);
    McpSyncClient client = mockClientNamed("late");
    when(context.getBeansOfType(McpSyncClient.class)).thenReturn(Map.of());
    when(context.getBeanNamesForType(
            ResolvableType.forClassWithGenerics(List.class, McpSyncClient.class)))
        .thenReturn(new String[0])
        .thenReturn(new String[] {"mcpSyncClients"});
    when(context.getBean("mcpSyncClients", List.class)).thenReturn(List.of(client));
    McpPlugin plugin = new McpPlugin();
    plugin.setApplicationContext(context);
    Worker worker = mock(Worker.class);

    plugin.initializeWorker("mcp-tq", worker);
    verifyNoInteractions(worker);
    plugin.afterSingletonsInstantiated();

    ArgumentCaptor<Object> captor = ArgumentCaptor.forClass(Object.class);
    verify(worker).registerActivitiesImplementations(captor.capture());
    McpClientActivityImpl activity = (McpClientActivityImpl) captor.getValue();
    assertEquals(Set.of("late"), activity.getClientInfo().keySet());
  }

  @Configuration(proxyBeanMethods = false)
  static class ClientLists {
    @Bean
    List<McpSyncClient> mcpSyncClients() {
      return List.of(mockClientNamed("alpha"), mockClientNamed("beta"));
    }

    @Bean
    List<McpSyncClient> otherMcpClients() {
      return List.of(mockClientNamed("gamma"));
    }

    @Bean
    List<String> unrelatedList() {
      return List.of("not an MCP client");
    }
  }

  @Test
  void discoversMcpClientBeansByType() {
    McpSyncClient clientA = mockClientNamed("alpha");
    McpSyncClient clientB = mockClientNamed("beta");

    // Spring's getBeansOfType keeps insertion order via LinkedHashMap; use that for determinism.
    Map<String, McpSyncClient> beans = new LinkedHashMap<>();
    beans.put("mcpClientAlpha", clientA);
    beans.put("mcpClientBeta", clientB);

    ApplicationContext ctx = contextWithoutClientLists();
    when(ctx.getBeansOfType(McpSyncClient.class)).thenReturn(beans);

    McpPlugin plugin = new McpPlugin();
    plugin.setApplicationContext(ctx);

    Worker worker = mock(Worker.class);
    plugin.initializeWorker("mcp-tq", worker);

    ArgumentCaptor<Object> captor = ArgumentCaptor.forClass(Object.class);
    verify(worker, atLeastOnce()).registerActivitiesImplementations(captor.capture());
    Object registered = captor.getValue();
    assertEquals(McpClientActivityImpl.class, registered.getClass());
  }

  @Test
  void twoMcpBeans_duplicateClientInfoNames_throws() {
    // Two distinct beans that both report the same clientInfo().name() — the activity impl
    // has to reject this because it keys its internal client map by that name.
    McpSyncClient clientA = mockClientNamed("shared");
    McpSyncClient clientB = mockClientNamed("shared");

    Map<String, McpSyncClient> beans = new LinkedHashMap<>();
    beans.put("mcpClientA", clientA);
    beans.put("mcpClientB", clientB);

    ApplicationContext ctx = contextWithoutClientLists();
    when(ctx.getBeansOfType(McpSyncClient.class)).thenReturn(beans);

    McpPlugin plugin = new McpPlugin();
    plugin.setApplicationContext(ctx);

    IllegalArgumentException thrown =
        assertThrows(
            IllegalArgumentException.class,
            () -> plugin.initializeWorker("mcp-tq", mock(Worker.class)));
    assertTrue(
        thrown.getMessage().contains("shared"),
        "expected duplicate name in message, got: " + thrown.getMessage());
  }

  @Test
  void noMcpBeans_defersWorker_thenClearsAfterSingletonsInstantiated() {
    ApplicationContext ctx = contextWithoutClientLists();
    when(ctx.getBeansOfType(McpSyncClient.class)).thenReturn(Map.of());

    McpPlugin plugin = new McpPlugin();
    plugin.setApplicationContext(ctx);

    Worker worker = mock(Worker.class);
    plugin.initializeWorker("mcp-tq", worker);

    // No beans → nothing registered yet, worker queued for deferred attempt.
    verifyNoInteractions(worker);

    plugin.afterSingletonsInstantiated();

    // Still no beans — the deferred attempt also finds nothing and doesn't crash.
    verify(worker, org.mockito.Mockito.never()).registerActivitiesImplementations((Object[]) any());
  }

  @Test
  void beansAppearLate_registeredViaAfterSingletonsInstantiated() {
    ApplicationContext ctx = contextWithoutClientLists();
    // First lookup returns empty (Spring AI MCP bean hasn't been created yet when
    // initializeWorker runs).
    when(ctx.getBeansOfType(McpSyncClient.class))
        .thenReturn(Map.of())
        .thenReturn(Map.of("mcpClient", mockClientNamed("late")));

    McpPlugin plugin = new McpPlugin();
    plugin.setApplicationContext(ctx);

    Worker worker = mock(Worker.class);
    plugin.initializeWorker("mcp-tq", worker);
    verifyNoInteractions(worker);

    plugin.afterSingletonsInstantiated();

    ArgumentCaptor<Object> captor = ArgumentCaptor.forClass(Object.class);
    verify(worker, atLeastOnce()).registerActivitiesImplementations(captor.capture());
    assertEquals(McpClientActivityImpl.class, captor.getValue().getClass());
  }

  private static ApplicationContext contextWithoutClientLists() {
    ApplicationContext context = mock(ApplicationContext.class);
    when(context.getBeanNamesForType(any(ResolvableType.class))).thenReturn(new String[0]);
    return context;
  }

  private static McpSyncClient mockClientNamed(String name) {
    McpSyncClient client = mock(McpSyncClient.class);
    McpSchema.Implementation info = McpSchema.Implementation.builder(name, "1.0.0").build();
    when(client.getClientInfo()).thenReturn(info);
    return client;
  }
}
