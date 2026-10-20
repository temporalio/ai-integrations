package main

import (
	"fmt"
	"github.com/bazelbuild/rules_go/go/runfiles"
	"os"
	"path/filepath"
	"strings"
)

var goRunfile string

func main() {
	if goRunfile != "" {
		path, err := runfiles.Rlocation(goRunfile)
		if err != nil {
			panic(err)
		}
		os.Setenv("AI_GO", path)
		os.Setenv("GOROOT", filepath.Dir(filepath.Dir(path)))
		os.Setenv("PATH", filepath.Dir(path)+string(os.PathListSeparator)+os.Getenv("PATH"))
	}

	pos, o, err := parseArgs(os.Args[1:])
	if err == nil && o["repo-root"] != "" {
		err = os.Setenv("BUILD_WORKSPACE_DIRECTORY", o["repo-root"])
	}
	for _, key := range []string{"dist", "output", "registry-json"} {
		if o[key] != "" && !filepath.IsAbs(o[key]) {
			o[key] = filepath.Join(workspace(), o[key])
		}
	}
	mode := strings.TrimSuffix(filepath.Base(os.Args[0]), ".exe")
	if len(pos) > 0 && (pos[0] == "ci" || pos[0] == "release" || pos[0] == "dev") {
		mode, pos = pos[0], pos[1:]
	}
	if err == nil {
		switch mode {
		case "release":
			err = releaseMain(pos, o)
		case "ci":
			err = ciMain(pos, o)
		case "lint":
			err = lintMain(pos, o)
		case "dev":
			err = devMain(pos, o)
		default:
			err = fmt.Errorf("unknown Go tooling executable %s", mode)
		}
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
