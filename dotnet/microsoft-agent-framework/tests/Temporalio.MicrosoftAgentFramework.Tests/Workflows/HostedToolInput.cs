namespace Temporalio.MicrosoftAgentFramework.Tests.Workflows;

public sealed record HostedToolInput(string Prompt, bool WorkflowSideHeaders = false);
