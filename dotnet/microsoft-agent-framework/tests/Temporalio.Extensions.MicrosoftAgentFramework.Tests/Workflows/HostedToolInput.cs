namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests.Workflows;

public sealed record HostedToolInput(string Prompt, bool WorkflowSideHeaders = false);
