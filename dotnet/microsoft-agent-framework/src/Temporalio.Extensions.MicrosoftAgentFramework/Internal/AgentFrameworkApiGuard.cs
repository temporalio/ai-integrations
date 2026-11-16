using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Temporalio.Exceptions;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Internal;

/// <summary>
/// Checks at plugin construction that the installed <c>Microsoft.Agents.AI</c> still has the
/// experimental (<c>MAAI001</c>) provider context constructors the loop calls. The package
/// reference allows every release below the next major, so a user can take a minor update without
/// waiting for a plugin release; a release that changes one of these constructors then fails here,
/// with a message that names the tested version, instead of with <c>MissingMethodException</c>
/// in the middle of a run.
/// </summary>
internal static class AgentFrameworkApiGuard
{
    /// <summary>The <c>Microsoft.Agents.AI</c> version the plugin's test suite ran against.</summary>
    public const string TestedVersion = "1.26.0";

    /// <summary>The range the package reference accepts.</summary>
    public const string AllowedRange = "1.26.0 up to, and not including, 2.0.0";

    private static readonly IReadOnlyList<(Type Type, Type[] Parameters)> Constructors =
    [
        (typeof(ChatHistoryProvider.InvokingContext), [typeof(AIAgent), typeof(AgentSession), typeof(IEnumerable<ChatMessage>)]),
        (typeof(ChatHistoryProvider.InvokedContext), [typeof(AIAgent), typeof(AgentSession), typeof(IEnumerable<ChatMessage>), typeof(IEnumerable<ChatMessage>)]),
        (typeof(ChatHistoryProvider.InvokedContext), [typeof(AIAgent), typeof(AgentSession), typeof(IEnumerable<ChatMessage>), typeof(Exception)]),
        (typeof(AIContextProvider.InvokingContext), [typeof(AIAgent), typeof(AgentSession), typeof(AIContext)]),
        (typeof(AIContextProvider.InvokedContext), [typeof(AIAgent), typeof(AgentSession), typeof(IEnumerable<ChatMessage>), typeof(IEnumerable<ChatMessage>)]),
        (typeof(AIContextProvider.InvokedContext), [typeof(AIAgent), typeof(AgentSession), typeof(IEnumerable<ChatMessage>), typeof(Exception)]),
    ];

    /// <summary>Checks the constructors the plugin calls.</summary>
    public static void Check() => Check(Constructors);

    /// <summary>
    /// Checks that each listed constructor exists.
    /// </summary>
    /// <param name="constructors">Types and the parameter types of the constructor each must have.</param>
    /// <exception cref="ApplicationFailureException">
    /// With <see cref="MicrosoftAgentFrameworkErrorTypes.ConfigurationError"/> when one is missing.
    /// </exception>
    internal static void Check(IEnumerable<(Type Type, Type[] Parameters)> constructors)
    {
        var missing = constructors
            .Where(constructor => constructor.Type.GetConstructor(constructor.Parameters) is null)
            .Select(constructor => $"{constructor.Type.FullName}({string.Join(", ", constructor.Parameters.Select(parameter => parameter.Name))})")
            .ToList();
        if (missing.Count == 0)
        {
            return;
        }

        var installed = typeof(AIAgent).Assembly.GetName().Version?.ToString(3) ?? "unknown";
        throw new ApplicationFailureException(
            $"Microsoft.Agents.AI {installed} does not have the experimental constructor(s) this plugin calls: {string.Join("; ", missing)}. " +
            $"The plugin was tested with {TestedVersion} and its package reference accepts {AllowedRange}; pin Microsoft.Agents.AI to a tested version or update the plugin.",
            errorType: MicrosoftAgentFrameworkErrorTypes.ConfigurationError,
            nonRetryable: true);
    }
}
