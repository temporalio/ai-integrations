package main

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"golang.org/x/mod/semver"
)

var firstVersion = map[string]string{"generally-available": "1.0.0", "public-preview": "0.1.0", "pre-release": "0.0.1"}

func validVersion(raw string) error {
	if !semver.IsValid("v"+raw) || semver.Canonical("v"+raw) != "v"+raw || strings.Contains(raw, "+") {
		return fmt.Errorf("Go version %q must be canonical SemVer without build metadata", raw)
	}
	return nil
}
func checkPolicy(raw, maturity string, published []string) error {
	if err := validVersion(raw); err != nil {
		return err
	}
	first, ok := firstVersion[maturity]
	if !ok {
		return errors.New("unknown maturity")
	}
	if len(published) == 0 {
		if strings.SplitN(raw, "-", 2)[0] != first {
			return fmt.Errorf("first release must be %s", first)
		}
		return nil
	}
	for _, v := range published {
		if err := validVersion(v); err != nil {
			return err
		}
		if semver.Compare("v"+raw, "v"+v) < 0 {
			return fmt.Errorf("version is older than published %s", v)
		}
	}
	return nil
}
func parseTag(tag string) (map[string]string, error) {
	match := regexp.MustCompile(`^go/([a-z0-9][a-z0-9_.-]*)/v(.+)$`).FindStringSubmatch(tag)
	if match == nil || strings.Contains(match[1], "..") {
		return nil, errors.New("expected go/<plugin>/v<version>")
	}
	if err := validVersion(match[2]); err != nil {
		return nil, err
	}
	if semver.Major("v"+match[2]) != "v0" && semver.Major("v"+match[2]) != "v1" {
		return nil, errors.New("Go v2+ requires a module-path cutover")
	}
	return map[string]string{"tag": tag, "language": "go", "plugin": match[1], "plugin_dir": "go/" + match[1], "version": match[2], "prerelease": strconv.FormatBool(semver.Prerelease("v"+match[2]) != "")}, nil
}
func fetchBytes(address string, missing bool) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", address, nil)
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("registry unreachable; refusing to guess: %w", err)
	}
	defer resp.Body.Close()
	if missing && (resp.StatusCode == 404 || resp.StatusCode == 410) {
		return nil, nil
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("HTTP %d; refusing to guess", resp.StatusCode)
	}
	return io.ReadAll(resp.Body)
}
func escapeModule(coordinate string) string {
	var b strings.Builder
	for _, c := range coordinate {
		if c >= 'A' && c <= 'Z' {
			b.WriteByte('!')
			b.WriteRune(c + ('a' - 'A'))
		} else {
			b.WriteRune(c)
		}
	}
	return b.String()
}
func publishedVersions(coordinate, local string) ([]string, error) {
	var data []byte
	var err error
	if local != "" {
		data, err = os.ReadFile(local)
		if os.IsNotExist(err) {
			return nil, nil
		}
	} else {
		data, err = fetchBytes("https://proxy.golang.org/"+escapeModule(coordinate)+"/@v/list", true)
	}
	if err != nil {
		return nil, err
	}
	var versions []string
	for _, line := range strings.Fields(string(data)) {
		if !strings.HasPrefix(line, "v") {
			return nil, errors.New("invalid Go proxy version")
		}
		raw := line[1:]
		if err := validVersion(raw); err != nil {
			return nil, err
		}
		versions = append(versions, raw)
	}
	return versions, nil
}

var metaTag = regexp.MustCompile(`(?is)<meta\s+[^>]*>`)
var htmlAttribute = regexp.MustCompile(`(?is)([a-z-]+)\s*=\s*["']([^"']*)["']`)

func verifyVanityHTML(data []byte, coordinate, repo, dir string) error {
	for _, tag := range metaTag.FindAll(data, -1) {
		attrs := map[string]string{}
		for _, pair := range htmlAttribute.FindAllSubmatch(tag, -1) {
			attrs[strings.ToLower(string(pair[1]))] = string(pair[2])
		}
		if attrs["name"] != "go-import" {
			continue
		}
		fields := strings.Fields(attrs["content"])
		if len(fields) == 4 && fields[0] == coordinate && fields[1] == "git" && strings.TrimSuffix(fields[2], ".git") == "https://github.com/"+repo && fields[3] == dir {
			return nil
		}
	}
	return errors.New("Go vanity mapping must identify this repository and exact plugin subdirectory")
}
func verifyZip(root, dir, coordinate, version, archive string) error {
	data, err := run(root, os.Environ(), "git", "ls-files", "-z", "--", dir)
	if err != nil {
		return err
	}
	files := strings.Split(string(data), "\x00")
	module := filepath.Join(root, dir)
	nested := []string{}
	for _, file := range files {
		if file != "" && filepath.Base(file) == "go.mod" && filepath.Dir(filepath.Join(root, file)) != module {
			nested = append(nested, filepath.Dir(filepath.Join(root, file))+string(os.PathSeparator))
		}
	}
	expected := map[string][]byte{}
	for _, file := range files {
		if file == "" {
			continue
		}
		path := filepath.Join(root, file)
		info, err := os.Lstat(path)
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			continue
		}
		relative, _ := filepath.Rel(module, path)
		skip := false
		for _, part := range strings.Split(filepath.ToSlash(filepath.Dir(relative)), "/") {
			if part == "vendor" || part == ".git" || part == ".hg" || part == ".svn" || part == ".bzr" {
				skip = true
			}
		}
		for _, prefix := range nested {
			if strings.HasPrefix(path, prefix) {
				skip = true
			}
		}
		if skip {
			continue
		}
		expected[filepath.ToSlash(relative)], err = os.ReadFile(path)
		if err != nil {
			return err
		}
	}
	z, err := zip.OpenReader(archive)
	if err != nil {
		return err
	}
	defer z.Close()
	prefix := coordinate + "@v" + version + "/"
	seen := map[string]bool{}
	for _, file := range z.File {
		if file.FileInfo().IsDir() {
			continue
		}
		if !strings.HasPrefix(file.Name, prefix) || seen[file.Name] {
			return errors.New("duplicate archive file or unexpected module prefix")
		}
		seen[file.Name] = true
		name := strings.TrimPrefix(file.Name, prefix)
		want, ok := expected[name]
		if !ok {
			return fmt.Errorf("unexpected module file %s", name)
		}
		r, err := file.Open()
		if err != nil {
			return err
		}
		got, err := io.ReadAll(r)
		r.Close()
		if err != nil {
			return err
		}
		if !bytes.Equal(got, want) {
			return fmt.Errorf("module bytes differ: %s", name)
		}
		delete(expected, name)
	}
	if len(expected) > 0 {
		return errors.New("public module is missing tested source files")
	}
	return nil
}
func verifyModule(dir, coordinate, version string, attempts int, delay time.Duration) error {
	if attempts < 1 || delay < 0 {
		return errors.New("invalid retry settings")
	}
	tmp, err := os.MkdirTemp("", "go-release-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	envMap := map[string]string{}
	for _, entry := range os.Environ() {
		pair := strings.SplitN(entry, "=", 2)
		envMap[pair[0]] = pair[1]
	}
	for key, value := range map[string]string{"GOPROXY": "https://proxy.golang.org", "GOSUMDB": "sum.golang.org", "GOPRIVATE": "", "GONOPROXY": "", "GONOSUMDB": "", "GOWORK": "off", "GOTOOLCHAIN": "local", "GOFLAGS": "", "GOMODCACHE": filepath.Join(tmp, "modcache")} {
		envMap[key] = value
	}
	env := []string{}
	for key, value := range envMap {
		env = append(env, key+"="+value)
	}
	query := coordinate + "@v" + version
	var data []byte
	for attempt := 0; attempt < attempts; attempt++ {
		data, err = run(tmp, env, "go", "mod", "download", "-json", query)
		if err == nil {
			break
		}
		if attempt+1 < attempts {
			time.Sleep(delay)
		}
	}
	if err != nil {
		return err
	}
	var result struct{ Path, Version, Error, Sum, GoModSum, Zip, GoMod string }
	if err = json.Unmarshal(data, &result); err != nil {
		return err
	}
	if result.Path != coordinate || result.Version != "v"+version || result.Error != "" || !strings.HasPrefix(result.Sum, "h1:") || !strings.HasPrefix(result.GoModSum, "h1:") {
		return errors.New("public module identity or checksum verification differs")
	}
	if err = verifyZip(workspace(), dir, coordinate, version, result.Zip); err != nil {
		return err
	}
	served, err := os.ReadFile(result.GoMod)
	if err != nil {
		return err
	}
	tested, err := os.ReadFile(filepath.Join(workspace(), dir, "go.mod"))
	if err != nil {
		return err
	}
	if !bytes.Equal(served, tested) {
		return errors.New("public go.mod differs")
	}
	if err = os.WriteFile(filepath.Join(tmp, "main.go"), []byte("package main\nimport _ "+strconv.Quote(coordinate)+"\nfunc main() {}\n"), 0600); err != nil {
		return err
	}
	for _, args := range [][]string{{"mod", "init", "release-smoke"}, {"get", query}, {"mod", "tidy"}, {"mod", "verify"}, {"build", "-mod=readonly", "./..."}} {
		if _, err = run(tmp, env, "go", args...); err != nil {
			return err
		}
	}
	return nil
}
func releaseMain(pos []string, o options) error {
	if len(pos) == 0 {
		return errors.New("missing release operation")
	}
	action := pos[0]
	repo := o["repo"]
	if repo == "" {
		repo = repository()
	}
	if action == "parse-tag" {
		if len(pos) != 2 {
			return errors.New("missing release tag")
		}
		out, err := parseTag(pos[1])
		if err != nil {
			return err
		}
		meta, err := loadMeta(out["plugin_dir"])
		if err != nil {
			return err
		}
		coordinate := "go.temporal.io/" + out["plugin"]
		if meta.Plugin.Name != out["plugin"] || meta.Plugin.Language != "go" || meta.Plugin.Registry != "goproxy" || meta.Plugin.Coordinate != coordinate {
			return errors.New("Go tag and plugin metadata differ")
		}
		mod, err := os.ReadFile(filepath.Join(workspace(), out["plugin_dir"], "go.mod"))
		if err != nil {
			return err
		}
		match := regexp.MustCompile(`(?m)^module\s+"?([^\s"]+)"?`).FindSubmatch(mod)
		if len(match) != 2 || string(match[1]) != coordinate {
			return errors.New("go.mod coordinate differs")
		}
		if _, ok := firstVersion[meta.Plugin.Maturity]; !ok {
			return errors.New("unknown maturity")
		}
		out["coordinate"] = coordinate
		out["root_api"] = meta.Plugin.RootAPI
		out["smoke_imports"] = strings.Join(meta.Smoke.Imports, ",")
		out["maturity"] = meta.Plugin.Maturity
		out["github_prerelease"] = strconv.FormatBool(out["prerelease"] == "true" || meta.Plugin.Maturity == "pre-release")
		if len(meta.CI.Runtimes) == 0 {
			return errors.New("missing runtime versions")
		}
		out["go_runtime"] = meta.CI.Runtimes[len(meta.CI.Runtimes)-1]
		return output(out, o["github-output"])
	}
	if action == "check-version-policy" {
		meta, err := loadMeta(o["plugin-dir"])
		if err != nil {
			return err
		}
		published, err := publishedVersions(meta.Plugin.Coordinate, o["registry-json"])
		if err != nil {
			return err
		}
		if err = checkPolicy(o["version"], meta.Plugin.Maturity, published); err != nil {
			return err
		}
		return nil
	}
	if action == "check-go-vanity" {
		meta, err := loadMeta(o["plugin-dir"])
		if err != nil {
			return err
		}
		data, err := fetchBytes("https://"+meta.Plugin.Coordinate+"?go-get=1", false)
		if err != nil {
			return err
		}
		return verifyVanityHTML(data, meta.Plugin.Coordinate, repo, o["plugin-dir"])
	}
	if action == "verify-go-module" {
		meta, err := loadMeta(o["plugin-dir"])
		if err != nil {
			return err
		}
		if err = validVersion(o["version"]); err != nil {
			return err
		}
		attempts := 10
		delay := 30
		if o["attempts"] != "" {
			attempts, err = strconv.Atoi(o["attempts"])
			if err != nil {
				return err
			}
		}
		if o["delay"] != "" {
			delay, err = strconv.Atoi(o["delay"])
			if err != nil {
				return err
			}
		}
		return verifyModule(o["plugin-dir"], meta.Plugin.Coordinate, o["version"], attempts, time.Duration(delay)*time.Second)
	}
	if action == "check-published-release" {
		parsed, err := parseTag(o["tag"])
		if err != nil {
			return err
		}
		meta, err := loadMeta(parsed["plugin_dir"])
		if err != nil {
			return err
		}
		if _, ok := firstVersion[meta.Plugin.Maturity]; !ok {
			return errors.New("unknown maturity")
		}
		data, err := gh("repos/" + repo + "/releases/tags/" + url.PathEscape(o["tag"]))
		if err != nil {
			return err
		}
		var release struct {
			Tag        string `json:"tag_name"`
			Draft      *bool
			Prerelease *bool
			URL        string `json:"html_url"`
		}
		if err = json.Unmarshal(data, &release); err != nil {
			return err
		}
		if release.Tag != o["tag"] || release.Draft == nil || *release.Draft || release.Prerelease == nil || *release.Prerelease != (parsed["prerelease"] == "true" || meta.Plugin.Maturity == "pre-release") || release.URL == "" {
			return errors.New("published GitHub release differs")
		}
		return output(map[string]string{"url": release.URL}, o["github-output"])
	}
	if action == "release-notes" {
		parsed, err := parseTag(o["tag"])
		if err != nil {
			return err
		}
		meta, err := loadMeta(o["plugin-dir"])
		if err != nil {
			return err
		}
		tags, err := git("tag", "--list", "go/"+parsed["plugin"]+"/v*")
		if err != nil {
			return err
		}
		previous := []string{}
		for _, tag := range strings.Fields(tags) {
			parts := strings.Split(tag, "/v")
			v := parts[len(parts)-1]
			if validVersion(v) == nil && semver.Compare("v"+v, "v"+parsed["version"]) < 0 {
				previous = append(previous, tag)
			}
		}
		sort.Slice(previous, func(i, j int) bool {
			a, b := strings.Split(previous[i], "/v"), strings.Split(previous[j], "/v")
			return semver.Compare("v"+a[len(a)-1], "v"+b[len(b)-1]) > 0
		})
		ref := o["ref"]
		if ref == "" {
			ref = o["tag"]
		}
		rangeRef := ref
		var notes strings.Builder
		if len(previous) > 0 {
			rangeRef = previous[0] + ".." + ref
		} else {
			fmt.Fprintf(&notes, "First standalone release of `%s`.\n\n", meta.Plugin.Coordinate)
		}
		notes.WriteString("## What's changed\n\n")
		log, err := git("log", "--no-decorate", "--format=%h%x1f%s", rangeRef, "--", o["plugin-dir"])
		if err != nil {
			return err
		}
		for _, line := range strings.Split(log, "\n") {
			if line == "" {
				continue
			}
			parts := strings.SplitN(line, "\x1f", 2)
			fmt.Fprintf(&notes, "- %s (%s)\n", parts[1], parts[0])
		}
		if len(previous) > 0 {
			fmt.Fprintf(&notes, "\n**Full changelog**: https://github.com/%s/compare/%s...%s\n", repo, previous[0], o["tag"])
		} else {
			fmt.Fprintf(&notes, "\n**Source**: https://github.com/%s/tree/refs/tags/%s/%s\n", repo, o["tag"], o["plugin-dir"])
		}
		if parsed["prerelease"] == "true" {
			fmt.Fprintf(&notes, "\n## Pre-release notes\n\nInstall this Go pre-release with `go get %s@v%s`.\n", meta.Plugin.Coordinate, parsed["version"])
		}
		return os.WriteFile(o["output"], []byte(notes.String()), 0600)
	}
	return fmt.Errorf("unknown Go release operation %q", action)
}

// Keep exec imported here to assert Go consumers use an external toolchain rather than a local checkout.
var _ = exec.Command
