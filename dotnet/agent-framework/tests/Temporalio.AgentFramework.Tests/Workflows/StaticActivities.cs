using Temporalio.Activities;

namespace Temporalio.AgentFramework.Tests.Workflows;

public static class StaticActivities
{
    [Activity]
    public static string LookupTimeZone(string city) => $"{city}/UTC-5";
}
