using System.ComponentModel;
using Temporalio.Activities;

namespace Temporalio.MicrosoftAgentFramework.Tests.Workflows;

public sealed class WeatherActivities
{
    [Activity]
    [Description("Gets the weather for a city")]
    public Task<string> GetWeatherAsync([Description("City name")] string city) =>
        Task.FromResult($"sunny in {city} (activity)");

    [Activity]
    public Task<WeatherReport> GetReportAsync(string city, int days) =>
        Task.FromResult(new WeatherReport(city, $"sunny for {days} days"));

    [Activity]
    public Task<string> SendEmailAsync(string to) => Task.FromResult($"email sent to {to}");

    // Heartbeats until the Workflow cancels it, so a cancellation test can see the cancel reach the Activity.
    [Activity]
    public async Task<string> SlowWeatherAsync(string city)
    {
        var context = ActivityExecutionContext.Current;
        while (true)
        {
            context.Heartbeat();
            await Task.Delay(100, context.CancellationToken);
        }
    }

    // The trailing optional token stays out of the tool schema and of the call; the SDK fills its default.
    [Activity]
    [Description("Gets the weather alerts for a city")]
    public Task<string> GetAlertsAsync([Description("City name")] string city, CancellationToken cancellationToken = default) =>
        Task.FromResult(cancellationToken.CanBeCanceled ? $"alerts for {city} (cancellable)" : $"no alerts for {city}");
}
