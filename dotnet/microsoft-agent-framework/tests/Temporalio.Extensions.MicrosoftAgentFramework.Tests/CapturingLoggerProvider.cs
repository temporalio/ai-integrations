using Microsoft.Extensions.Logging;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests;

/// <summary>Collects log lines so tests can assert on Workflow warnings.</summary>
public sealed class CapturingLoggerProvider : ILoggerProvider, ILoggerFactory
{
    private readonly List<(LogLevel Level, string Message)> entries = [];
    private readonly object sync = new();

    public IReadOnlyList<(LogLevel Level, string Message)> Entries
    {
        get
        {
            lock (sync)
            {
                return entries.ToList();
            }
        }
    }

    public ILogger CreateLogger(string categoryName) => new Logger(this);

    public void AddProvider(ILoggerProvider provider)
    {
    }

    public void Dispose()
    {
    }

    private void Add(LogLevel level, string message)
    {
        lock (sync)
        {
            entries.Add((level, message));
        }
    }

    private sealed class Logger : ILogger
    {
        private readonly CapturingLoggerProvider owner;

        public Logger(CapturingLoggerProvider owner) => this.owner = owner;

        public IDisposable? BeginScope<TState>(TState state)
            where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter) =>
            owner.Add(logLevel, formatter(state, exception));
    }
}
