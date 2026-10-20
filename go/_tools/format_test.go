package main

import (
	"bytes"
	"go/format"
	"os"
	"strings"
	"testing"

	"github.com/bazelbuild/rules_go/go/runfiles"
)

func TestFormatting(t *testing.T) {
	for _, name := range strings.Fields(os.Getenv("GO_PLUGIN_FILES")) {
		path, err := runfiles.Rlocation(name)
		if err != nil {
			t.Fatal(err)
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		formatted, err := format.Source(raw)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(formatted, raw) {
			t.Errorf("gofmt differs: %s", name)
		}
	}
}
