using System.Reflection;

namespace Temporalio.Extensions.MicrosoftAgentFramework.Tests;

public class ProvenanceTests
{
    [Fact]
    public void AssemblyUnderTest_ComesFromThisCheckout()
    {
        var assembly = typeof(MicrosoftAgentFrameworkPlugin).Assembly;
        Assert.Equal(PluginMeta.Coordinate, assembly.GetName().Name);
        Assert.StartsWith(Path.GetFullPath(AppContext.BaseDirectory), Path.GetFullPath(assembly.Location), StringComparison.Ordinal);

        var informational = assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
        Assert.NotNull(informational);
        var testInformational = typeof(ProvenanceTests).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
        Assert.Equal(CommitOf(testInformational), CommitOf(informational));
        if (Environment.GetEnvironmentVariable("GITHUB_SHA") is { Length: > 0 } sha)
        {
            Assert.EndsWith("+" + sha, informational, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void PluginToml_MatchesAssembly()
    {
        Assert.Equal("microsoft-agent-framework", PluginMeta.Name);
        Assert.Equal("Temporalio.Extensions.MicrosoftAgentFramework", PluginMeta.Coordinate);
        Assert.Equal(typeof(MicrosoftAgentFrameworkPlugin).Namespace, PluginMeta.RootApi);
        var imports = PluginMeta.SmokeImports;
        Assert.NotEmpty(imports);
        foreach (var import in imports)
        {
            Assert.NotNull(Type.GetType($"{import}, {PluginMeta.Coordinate}", throwOnError: false));
        }
    }

    private static string CommitOf(string? informationalVersion)
    {
        var plus = informationalVersion?.IndexOf('+', StringComparison.Ordinal) ?? -1;
        return plus < 0 ? string.Empty : informationalVersion![(plus + 1)..];
    }
}
