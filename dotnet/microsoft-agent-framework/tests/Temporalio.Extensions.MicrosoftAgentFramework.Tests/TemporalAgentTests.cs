using Microsoft.Agents.AI;
using Temporalio.Activities;
using Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;
using Temporalio.Workflows;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests;

/// <summary>Construction-time checks that need no Workflow.</summary>
public class TemporalAgentTests
{
    [Fact]
    public void Providers_SharingAStateKey_AreRejected()
    {
        var options = new TemporalAgentOptions
        {
            AIContextProviders = [new KeyedContextProvider("memory"), new KeyedContextProvider("memory")],
        };
        var error = Assert.Throws<ArgumentException>(() => new TemporalAgent(TestPlugin.Model, options));
        Assert.Contains("share the session state key 'memory'", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Provider_UsingThePluginsStateKey_IsRejected()
    {
        var options = new TemporalAgentOptions { AIContextProviders = [new KeyedContextProvider("TemporalAgent.DeferredApprovalRequests")] };
        var error = Assert.Throws<ArgumentException>(() => new TemporalAgent(TestPlugin.Model, options));
        Assert.Contains("TemporalAgent.DeferredApprovalRequests", error.Message, StringComparison.Ordinal);
        Assert.Contains("held back behind an approval", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Providers_WithDistinctKeys_AreAccepted()
    {
        var options = new TemporalAgentOptions
        {
            AIContextProviders = [new KeyedContextProvider("memory"), new KeyedContextProvider("profile")],
            ChatHistoryProvider = new InMemoryChatHistoryProvider(),
        };
        var agent = new TemporalAgent(TestPlugin.Model, options);
        Assert.Equal(2, agent.AIContextProviders.Count);
    }

    [Fact]
    public void ActivityAsTool_TokenThatIsNotTrailingOptional_IsRejected()
    {
        var options = new ActivityOptions { StartToCloseTimeout = TimeSpan.FromSeconds(5) };
        var error = Assert.Throws<ArgumentException>(() => ActivityAsTool.Create(() => TokenFirstActivities.Lookup(default, default!), options));
        Assert.Contains("CancellationToken", error.Message, StringComparison.Ordinal);
        Assert.Contains("ActivityExecutionContext.Current.CancellationToken", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ActivityAsTool_TrailingOptionalToken_LeavesTheSchemaAndTheCall()
    {
        var options = new ActivityOptions { StartToCloseTimeout = TimeSpan.FromSeconds(5) };
        var tool = ActivityAsTool.Create((WeatherActivities a) => a.GetAlertsAsync(default!, default), options);
        var schema = tool.JsonSchema.GetRawText();
        Assert.Contains("\"city\"", schema, StringComparison.Ordinal);
        Assert.DoesNotContain("cancellationToken", schema, StringComparison.OrdinalIgnoreCase);
    }

    private static class TokenFirstActivities
    {
#pragma warning disable CA1068 // The test needs a token that is not the last parameter.
        [Activity]
        public static string Lookup(CancellationToken cancellationToken, string city) => city;
#pragma warning restore CA1068
    }
}
