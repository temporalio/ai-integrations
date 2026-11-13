using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Temporalio.AgentFramework.Tests;

/// <summary>Every C# block in the README compiles against the plugin and its dependencies.</summary>
public partial class ReadmeSnippetTests
{
    private const string Prelude = """
        global using System;
        global using System.Collections.Generic;
        global using System.IO;
        global using System.Linq;
        global using System.Net.Http;
        global using System.Threading;
        global using System.Threading.Tasks;
        """;

    public static IEnumerable<TheoryDataRow<int, string>> Blocks()
    {
        var readme = File.ReadAllText(PluginMeta.ReadmePath);
        var index = 0;
        foreach (Match match in FencePattern().Matches(readme))
        {
            yield return new TheoryDataRow<int, string>(index++, match.Groups[1].Value);
        }
    }

    [Fact]
    public void Readme_HasCodeBlocks() => Assert.True(Blocks().Count() >= 5);

    [Theory]
    [MemberData(nameof(Blocks))]
    public void Block_Compiles(int index, string code)
    {
        var token = TestContext.Current.CancellationToken;
        var tree = CSharpSyntaxTree.ParseText(code, new CSharpParseOptions(LanguageVersion.Latest), cancellationToken: token);
        var isProgram = tree.GetRoot(token).DescendantNodes().OfType<GlobalStatementSyntax>().Any();
        var compilation = CSharpCompilation.Create(
            $"ReadmeBlock{index}",
            [CSharpSyntaxTree.ParseText(Prelude, cancellationToken: token), tree],
            References(),
            new CSharpCompilationOptions(
                isProgram ? OutputKind.ConsoleApplication : OutputKind.DynamicallyLinkedLibrary,
                nullableContextOptions: NullableContextOptions.Enable));
        var errors = compilation.GetDiagnostics(token).Where(d => d.Severity == DiagnosticSeverity.Error).ToList();
        Assert.True(errors.Count == 0, $"README block {index} does not compile:\n{string.Join("\n", errors)}\n---\n{code}");
    }

    [Fact]
    public void Readme_QuickstartWorkflow_MatchesTestWorkflow()
    {
        var readme = File.ReadAllText(PluginMeta.ReadmePath);
        var source = File.ReadAllText(Path.Combine(PluginMeta.Root, "tests", "Temporalio.AgentFramework.Tests", "Workflows", "HelloWorkflow.workflow.cs"));
        var start = source.IndexOf("[Workflow]", StringComparison.Ordinal);
        var body = Normalize(source[start..]);
        Assert.Contains(body, Normalize(readme), StringComparison.Ordinal);
    }

    private static string Normalize(string text) =>
        string.Join("\n", text.Split('\n').Select(line => line.Trim()).Where(line => line.Length > 0));

    private static List<MetadataReference> References()
    {
        var paths = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var path in ((string)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES")!).Split(Path.PathSeparator))
        {
            paths[Path.GetFileName(path)] = path;
        }

        foreach (var path in Directory.GetFiles(AppContext.BaseDirectory, "*.dll"))
        {
            paths[Path.GetFileName(path)] = path;
        }

        return paths.Values
            .Where(path => !Path.GetFileName(path).StartsWith("Temporalio.AgentFramework.Tests", StringComparison.Ordinal))
            .Select(path => (MetadataReference)MetadataReference.CreateFromFile(path))
            .ToList();
    }

    [GeneratedRegex(@"```csharp\s*\n(.*?)```", RegexOptions.Singleline)]
    private static partial Regex FencePattern();
}
