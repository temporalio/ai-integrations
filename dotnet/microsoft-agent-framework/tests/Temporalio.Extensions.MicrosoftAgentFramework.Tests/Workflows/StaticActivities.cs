using Temporalio.Activities;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;

public static class StaticActivities
{
    [Activity]
    public static string LookupTimeZone(string city) => $"{city}/UTC-5";
}
