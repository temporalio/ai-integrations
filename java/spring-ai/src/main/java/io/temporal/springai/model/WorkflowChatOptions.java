package io.temporal.springai.model;

import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.model.tool.DefaultToolCallingChatOptions;
import org.springframework.ai.model.tool.ToolCallingChatOptions;

/** Keeps provider-specific options when ChatClient merges its customizer with model defaults. */
final class WorkflowChatOptions extends DefaultToolCallingChatOptions {
  WorkflowChatOptions() {
    super(null, null, null, null, null, null, null, null, null, null);
  }

  @Override
  public ToolCallingChatOptions.Builder<?> mutate() {
    return new ProviderBuilder();
  }

  private static final class ProviderBuilder
      extends DefaultToolCallingChatOptions.Builder<ProviderBuilder> {
    private ChatOptions.Builder<?> provider;

    @Override
    public ProviderBuilder combineWith(ChatOptions.Builder<?> other) {
      provider = other.clone();
      return super.combineWith(other);
    }

    @Override
    public ToolCallingChatOptions build() {
      ToolCallingChatOptions common = super.build();
      if (provider instanceof ToolCallingChatOptions.Builder<?> providerBuilder) {
        // combineWith appends callbacks, but common already includes the provider's list.
        return providerBuilder
            .clone()
            .combineWith(common.mutate())
            .toolCallbacks(common.getToolCallbacks())
            .build();
      }
      return common;
    }
  }
}
