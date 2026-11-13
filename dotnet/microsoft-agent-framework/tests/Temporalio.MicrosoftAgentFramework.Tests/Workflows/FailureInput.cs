namespace Temporalio.MicrosoftAgentFramework.Tests.Workflows;

public sealed record FailureInput(string Kind, string Model = TestPlugin.Model, string Prompt = "hello", int MaxAttempts = 3, int MaximumIterations = 40);
