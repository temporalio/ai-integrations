package googleadk

import (
	"runtime/debug"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestModuleVersion covers how Version reads this module's version from the
// build info of a program that depends on it.
func TestModuleVersion(t *testing.T) {
	const path = "go.temporal.io/googleadk"
	other := &debug.Module{Path: "example.com/other", Version: "v9.9.9"}
	for _, tc := range []struct {
		name string
		info *debug.BuildInfo
		ok   bool
		want string
	}{
		{"tagged", &debug.BuildInfo{Deps: []*debug.Module{other, {Path: path, Version: "v0.2.0"}}}, true, "0.2.0"},
		{"pseudo-version", &debug.BuildInfo{Deps: []*debug.Module{{Path: path, Version: "v0.0.0-20261102120000-abcdef123456"}}}, true, "0.0.0-20261102120000-abcdef123456"},
		{"module replace", &debug.BuildInfo{Deps: []*debug.Module{{Path: path, Version: "v0.2.0", Replace: &debug.Module{Path: "example.com/fork", Version: "v0.3.0"}}}}, true, "0.3.0"},
		{"directory replace", &debug.BuildInfo{Deps: []*debug.Module{{Path: path, Version: "v0.2.0", Replace: &debug.Module{Path: "../googleadk", Version: "(devel)"}}}}, true, ""},
		{"no entry", &debug.BuildInfo{Deps: []*debug.Module{other}}, true, ""},
		{"no build info", nil, false, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, moduleVersion(tc.info, tc.ok))
		})
	}
}
