using System.ComponentModel;
using Temporalio.Activities;

namespace Temporalio.AgentFramework.Tests.Workflows;

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
}
