package io.temporal.springai.model;

import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.model.tool.DefaultToolCallingChatOptions;
import org.springframework.ai.model.tool.StructuredOutputChatOptions;
import org.springframework.ai.model.tool.ToolCallingChatOptions;

/** Keeps caller builders unbuilt until they can be merged with the worker's model defaults. */
class WorkflowChatOptions extends DefaultToolCallingChatOptions {
  private final ChatOptions.Builder<?> provider;

  WorkflowChatOptions() {
    super(null, null, null, null, null, null, null, null, null, null);
    provider = null;
  }

  private WorkflowChatOptions(ToolCallingChatOptions common, ChatOptions.Builder<?> provider) {
    super(
        common.getToolCallbacks(),
        common.getToolContext(),
        common.getModel(),
        common.getFrequencyPenalty(),
        common.getMaxTokens(),
        common.getPresencePenalty(),
        common.getStopSequences(),
        common.getTemperature(),
        common.getTopK(),
        common.getTopP());
    this.provider = provider;
  }

  ChatOptions.Builder<?> activityOptions() {
    return provider != null ? provider.clone() : super.mutate();
  }

  @Override
  public ToolCallingChatOptions.Builder<?> mutate() {
    ProviderBuilder builder = new ProviderBuilder();
    builder.combineWith(super.mutate());
    builder.provider = provider != null ? provider.clone() : null;
    return builder;
  }

  private static final class ProviderBuilder
      extends DefaultToolCallingChatOptions.Builder<ProviderBuilder>
      implements StructuredOutputChatOptions.Builder<ProviderBuilder> {
    private ChatOptions.Builder<?> provider;

    @Override
    public ProviderBuilder clone() {
      ProviderBuilder copy = super.clone();
      copy.provider = provider != null ? provider.clone() : null;
      return copy;
    }

    @Override
    public ProviderBuilder outputSchema(String schema) {
      if (provider instanceof StructuredOutputChatOptions.Builder<?> structuredBuilder) {
        structuredBuilder.outputSchema(schema);
      }
      return this;
    }

    @Override
    public ProviderBuilder combineWith(ChatOptions.Builder<?> other) {
      ChatOptions.Builder<?> incoming = other;
      if (other instanceof ProviderBuilder workflowBuilder) {
        incoming = workflowBuilder.provider;
      }
      if (incoming != null) {
        if (provider == null) {
          provider = incoming.clone();
        } else {
          provider.combineWith(incoming);
        }
      }
      return super.combineWith(other);
    }

    @Override
    public ToolCallingChatOptions build() {
      ToolCallingChatOptions common = super.build();
      ChatOptions.Builder<?> merged = provider != null ? provider.clone() : common.mutate();
      if (provider != null) {
        // Common already includes provider lists. Copy them once after applying scalar overrides.
        merged.combineWith(common.mutate()).stopSequences(common.getStopSequences());
        if (merged instanceof ToolCallingChatOptions.Builder<?> toolBuilder) {
          toolBuilder.toolCallbacks(common.getToolCallbacks());
        }
      }
      return merged instanceof StructuredOutputChatOptions.Builder<?>
          ? new StructuredWorkflowChatOptions(common, merged)
          : new WorkflowChatOptions(common, merged);
    }
  }

  // Keep ChatClient's native structured-output advisor available when the caller's provider
  // supports it, while deferring the options build that would otherwise fill in model defaults.
  private static final class StructuredWorkflowChatOptions extends WorkflowChatOptions
      implements StructuredOutputChatOptions {
    private StructuredWorkflowChatOptions(
        ToolCallingChatOptions common, ChatOptions.Builder<?> provider) {
      super(common, provider);
    }

    @Override
    public String getOutputSchema() {
      return ((StructuredOutputChatOptions) activityOptions().build()).getOutputSchema();
    }

    @Override
    public ProviderBuilder mutate() {
      return (ProviderBuilder) super.mutate();
    }
  }
}
