package googleadk

import (
	"os"
	"testing"

	"github.com/stretchr/testify/require"
	"golang.org/x/mod/modfile"
	"golang.org/x/mod/semver"
)

// ADKAtLeast reports whether the tests build against ADK version v or later.
// CI tests the ADK version in go.mod and, every night, the newest one, so a
// test that pins ADK behavior that changed between them checks which one runs.
// Test binaries carry no dependency versions, so it reads go.mod, which
// -mod=readonly keeps equal to the versions in the build.
func ADKAtLeast(t *testing.T, v string) bool {
	t.Helper()
	data, err := os.ReadFile("go.mod")
	require.NoError(t, err)
	f, err := modfile.ParseLax("go.mod", data, nil)
	require.NoError(t, err)
	for _, r := range f.Require {
		if r.Mod.Path == "google.golang.org/adk/v2" {
			return semver.Compare(r.Mod.Version, v) >= 0
		}
	}
	t.Fatal("go.mod does not require ADK")
	return false
}
