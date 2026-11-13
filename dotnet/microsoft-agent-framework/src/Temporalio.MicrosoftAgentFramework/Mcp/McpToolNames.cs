using System.Text.RegularExpressions;

namespace Temporalio.MicrosoftAgentFramework.Mcp;

/// <summary>
/// Builds the names the model sees for MCP tools, the way Agent Framework's MCP tool builds them:
/// characters outside <c>A-Za-z0-9_.-</c> become <c>-</c>, and the prefix is joined to the name with
/// one <c>_</c>, after the separators at the end of the prefix and the start of the name are removed.
/// </summary>
internal static partial class McpToolNames
{
    private static readonly char[] Separators = ['_', '.', '-'];

    /// <summary>
    /// Builds the model-facing name.
    /// </summary>
    /// <param name="prefix">The prefix, or null or empty for none.</param>
    /// <param name="serverToolName">The name the server lists.</param>
    /// <returns>The name, for example <c>gh_tool</c> for <c>gh</c> or <c>gh_</c> and <c>tool</c>.</returns>
    public static string Build(string? prefix, string serverToolName)
    {
        ArgumentNullException.ThrowIfNull(serverToolName);
        var name = Normalize(serverToolName);
        if (string.IsNullOrEmpty(prefix))
        {
            return name;
        }

        var normalizedPrefix = Normalize(prefix).TrimEnd(Separators);
        if (normalizedPrefix.Length == 0)
        {
            return name;
        }

        var trimmedName = name.TrimStart(Separators);
        return trimmedName.Length == 0 ? normalizedPrefix : $"{normalizedPrefix}_{trimmedName}";
    }

    private static string Normalize(string value) => DisallowedCharacters().Replace(value, "-");

    [GeneratedRegex("[^A-Za-z0-9_.-]")]
    private static partial Regex DisallowedCharacters();
}
