using System.Text.RegularExpressions;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests;

/// <summary>Reads plugin.toml for provenance assertions.</summary>
public static partial class PluginMeta
{
    private static readonly Lazy<string> RootLazy = new(FindRoot);
    private static readonly Lazy<Dictionary<string, string>> ScalarsLazy = new(() => ParseScalars(File.ReadAllText(Path.Combine(Root, "plugin.toml"))));

    public static string Root => RootLazy.Value;

    public static string Name => ScalarsLazy.Value["name"];

    public static string Coordinate => ScalarsLazy.Value["coordinate"];

    public static string RootApi => ScalarsLazy.Value["root-api"];

    public static IReadOnlyList<string> SmokeImports
    {
        get
        {
            var toml = File.ReadAllText(Path.Combine(Root, "plugin.toml"));
            var match = ImportsPattern().Match(toml);
            return match.Success ? QuotedPattern().Matches(match.Groups[1].Value).Select(m => m.Groups[1].Value).ToList() : [];
        }
    }

    public static string ReadmePath => Path.Combine(Root, "README.md");

    // Walks up from the test binaries to the plugin directory, so the provenance and README tests need
    // the repository checkout around them; the test project cannot run from a copied bin/ directory.
    private static string FindRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "plugin.toml")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        throw new FileNotFoundException("plugin.toml not found above " + AppContext.BaseDirectory);
    }

    private static Dictionary<string, string> ParseScalars(string toml)
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (Match match in ScalarPattern().Matches(toml))
        {
            result[match.Groups[1].Value] = match.Groups[2].Value;
        }

        return result;
    }

    [GeneratedRegex(@"^([A-Za-z0-9_-]+)\s*=\s*""([^""]*)""", RegexOptions.Multiline)]
    private static partial Regex ScalarPattern();

    [GeneratedRegex(@"imports\s*=\s*\[([^\]]*)\]")]
    private static partial Regex ImportsPattern();

    [GeneratedRegex(@"""([^""]+)""")]
    private static partial Regex QuotedPattern();
}
