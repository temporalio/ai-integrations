// Repository tooling for Go integrations. This module is never published.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/pelletier/go-toml/v2"
)

type metadata struct {
	Plugin struct {
		Name, Language, Coordinate, Registry, Maturity string
		RootAPI                                        string `toml:"root-api"`
	} `toml:"plugin"`
	CI struct {
		Runtimes []string `toml:"runtime-versions"`
	} `toml:"ci"`
	Smoke struct {
		Imports []string `toml:"imports"`
	} `toml:"smoke"`
}
type options map[string]string

func parseArgs(args []string) ([]string, options, error) {
	var positional []string
	flags := options{}
	for i := 0; i < len(args); i++ {
		if !strings.HasPrefix(args[i], "--") {
			positional = append(positional, args[i])
			continue
		}
		key := strings.TrimPrefix(args[i], "--")
		if pair := strings.SplitN(key, "=", 2); len(pair) == 2 {
			flags[pair[0]] = pair[1]
			continue
		}
		if key == "dry-run" || key == "latest" {
			flags[key] = "true"
			continue
		}
		if i+1 == len(args) || strings.HasPrefix(args[i+1], "--") {
			return nil, nil, fmt.Errorf("missing value for --%s", key)
		}
		i++
		flags[key] = args[i]
	}
	return positional, flags, nil
}
func workspace() string {
	for _, name := range []string{"BUILD_WORKSPACE_DIRECTORY", "GITHUB_WORKSPACE"} {
		if value := os.Getenv(name); value != "" {
			return value
		}
	}
	path, _ := os.Getwd()
	return path
}
func repository() string {
	if repo := os.Getenv("GITHUB_REPOSITORY"); repo != "" {
		return repo
	}
	return "temporalio/ai-integrations"
}
func loadMeta(dir string) (metadata, error) {
	var meta metadata
	data, err := os.ReadFile(filepath.Join(workspace(), dir, "plugin.toml"))
	if err != nil {
		return meta, err
	}
	err = toml.Unmarshal(data, &meta)
	return meta, err
}
func output(values map[string]string, path string) error {
	if path == "" {
		path = os.Getenv("GITHUB_OUTPUT")
	}
	var lines strings.Builder
	for key, value := range values {
		if strings.ContainsAny(value, "\r\n") {
			return errors.New("output contains a line break")
		}
		fmt.Fprintf(&lines, "%s=%s\n", key, value)
	}
	fmt.Print(lines.String())
	if path != "" {
		f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
		if err != nil {
			return err
		}
		defer f.Close()
		_, err = f.WriteString(lines.String())
		return err
	}
	return nil
}
func run(dir string, env []string, name string, args ...string) ([]byte, error) {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	cmd.Env = env
	data, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("%s failed: %w\n%s", name, err, data)
	}
	return data, nil
}
func git(args ...string) (string, error) {
	data, err := run(workspace(), os.Environ(), "git", args...)
	return strings.TrimRight(string(data), "\n"), err
}
func gh(endpoint string, args ...string) (json.RawMessage, error) {
	data, err := run(workspace(), os.Environ(), "gh", append([]string{"api", endpoint}, args...)...)
	return data, err
}
