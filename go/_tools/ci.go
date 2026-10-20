package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/bazelbuild/rules_go/go/runfiles"
)

var metadataRunfile string

func metadataFile() (string, error) {
	if path := os.Getenv("REPO_METADATA"); path != "" {
		return path, nil
	}
	if metadataRunfile == "" {
		return "", errors.New("Bazel plugin metadata is missing from runfiles")
	}
	return runfiles.Rlocation(metadataRunfile)
}

func allMetadata() (map[string]json.RawMessage, error) {
	file, err := metadataFile()
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}
	var result map[string]json.RawMessage
	err = json.Unmarshal(data, &result)
	return result, err
}
func selected(files, names []string) []string {
	chosen := map[string]bool{}
	for _, file := range files {
		global := strings.HasPrefix(file, ".github/") || strings.HasPrefix(file, "bazel/") || file == "LICENSE" || file == "MODULE.bazel" || file == "MODULE.bazel.lock" || file == ".bazelrc" || file == ".bazelversion" || file == "REPO.bazel" || file == "BUILD.bazel"
		if global {
			return names
		}
		if !strings.HasPrefix(file, "go/") {
			continue
		}
		parts := strings.Split(file, "/")
		if len(parts) >= 3 {
			found := false
			for _, name := range names {
				if parts[1] == name {
					chosen[name] = true
					found = true
				}
			}
			if found {
				continue
			}
		}
		for _, name := range names {
			chosen[name] = true
		}
	}
	result := []string{}
	for name := range chosen {
		result = append(result, name)
	}
	sort.Strings(result)
	return result
}
func ciMain(pos []string, o options) error {
	if len(pos) == 0 {
		return errors.New("missing Go CI operation")
	}
	if pos[0] == "matrix" {
		all, err := allMetadata()
		if err != nil {
			return err
		}
		var meta struct{ Matrix json.RawMessage }
		if err = json.Unmarshal(all[o["plugin-dir"]], &meta); err != nil {
			return err
		}
		if len(meta.Matrix) == 0 {
			return errors.New("unknown plugin")
		}
		var compact bytes.Buffer
		if err := json.Compact(&compact, meta.Matrix); err != nil {
			return err
		}
		return output(map[string]string{"matrix": compact.String()}, o["github-output"])
	}
	if pos[0] == "select" {
		all, err := allMetadata()
		if err != nil {
			return err
		}
		names := []string{}
		for name := range all {
			if strings.HasPrefix(name, "go/") {
				names = append(names, strings.TrimPrefix(name, "go/"))
			}
		}
		sort.Strings(names)
		event := os.Getenv("GITHUB_EVENT_NAME")
		base := o["base"]
		if base == "" {
			base = "origin/" + os.Getenv("GITHUB_BASE_REF")
			if strings.HasSuffix(base, "/") {
				base += "main"
			}
		}
		head := "HEAD"
		spec := base + "..." + head
		if event == "merge_group" {
			base = os.Getenv("MERGE_GROUP_BASE_SHA")
			head = os.Getenv("MERGE_GROUP_HEAD_SHA")
			spec = base + " " + head
		}
		result := names
		if event == "pull_request" || event == "merge_group" || event == "" && o["base"] != "" {
			args := []string{"diff", "--name-only", "--no-renames"}
			args = append(args, strings.Fields(spec)...)
			files, e := git(args...)
			if e == nil {
				result = selected(strings.Split(files, "\n"), names)
			}
		}
		encoded, _ := json.Marshal(result)
		return output(map[string]string{"go": string(encoded)}, o["github-output"])
	}
	if pos[0] == "conventions" {
		entries, err := os.ReadDir(filepath.Join(workspace(), "go"))
		if err != nil {
			return err
		}
		license, err := os.ReadFile(filepath.Join(workspace(), "LICENSE"))
		if err != nil {
			return err
		}
		for _, entry := range entries {
			if !entry.IsDir() || strings.HasPrefix(entry.Name(), "_") || strings.HasPrefix(entry.Name(), ".") {
				continue
			}
			dir := "go/" + entry.Name()
			meta, err := loadMeta(dir)
			if err != nil {
				return err
			}
			if meta.Plugin.Name != entry.Name() || meta.Plugin.Language != "go" || meta.Plugin.Registry != "goproxy" || meta.Plugin.Coordinate != "go.temporal.io/"+entry.Name() {
				return errors.New("invalid Go plugin metadata")
			}
			for _, required := range []string{"go.mod", "go.sum", "BUILD.bazel", "README.md", "LICENSE"} {
				if _, err = os.Stat(filepath.Join(workspace(), dir, required)); err != nil {
					return err
				}
			}
			actual, err := os.ReadFile(filepath.Join(workspace(), dir, "LICENSE"))
			if err != nil {
				return err
			}
			if string(actual) != string(license) {
				return errors.New("Go plugin license differs")
			}
			if len(meta.CI.Runtimes) == 0 {
				return errors.New("Go plugin has no runtime matrix")
			}
		}
		return nil
	}
	if pos[0] == "nightly" {
		repo := o["repo"]
		if repo == "" {
			repo = repository()
		}
		runID := o["run-id"]
		if runID == "" {
			runID = os.Getenv("GITHUB_RUN_ID")
		}
		if runID == "" {
			return errors.New("nightly reporting requires a run ID")
		}
		data, err := gh("repos/"+repo+"/actions/runs/"+runID+"/jobs?per_page=100", "--paginate", "--slurp")
		if err != nil {
			return err
		}
		var pages []struct {
			Jobs []struct{ Name, Conclusion string }
		}
		if err = json.Unmarshal(data, &pages); err != nil {
			return err
		}
		seen := map[string]bool{}
		failures := map[string]bool{}
		pattern := regexp.MustCompile(`^Go \(([^)]+)\) / `)
		for _, page := range pages {
			for _, job := range page.Jobs {
				match := pattern.FindStringSubmatch(job.Name)
				if match == nil || job.Conclusion == "" {
					continue
				}
				key := "go/" + match[1]
				seen[key] = true
				if job.Conclusion == "failure" || job.Conclusion == "timed_out" {
					failures[key] = true
				}
			}
		}
		if o["dry-run"] == "true" {
			fmt.Println(failures)
			return nil
		}
		_, err = run(workspace(), os.Environ(), "gh", "label", "create", "nightly", "--repo", repo, "--force", "--color", "B60205", "--description", "Nightly checks with latest allowed dependencies failing")
		if err != nil {
			return err
		}
		issues, err := run(workspace(), os.Environ(), "gh", "issue", "list", "--repo", repo, "--label", "nightly", "--state", "open", "--limit", "200", "--json", "number,title")
		if err != nil {
			return err
		}
		var items []struct {
			Number int
			Title  string
		}
		if err = json.Unmarshal(issues, &items); err != nil {
			return err
		}
		byTitle := map[string]int{}
		for _, item := range items {
			byTitle[item.Title] = item.Number
		}
		link := "https://github.com/" + repo + "/actions/runs/" + runID
		for name := range seen {
			title := "nightly: " + name + " failing (latest)"
			id, exists := byTitle[title]
			args := []string{"issue"}
			if failures[name] {
				if exists {
					args = append(args, "comment", fmt.Sprint(id), "--repo", repo, "--body", "Still failing: "+link)
				} else {
					args = append(args, "create", "--repo", repo, "--title", title, "--label", "nightly", "--body", "Latest failed run: "+link)
				}
			} else if exists {
				args = append(args, "close", fmt.Sprint(id), "--repo", repo, "--comment", "Green again: "+link)
			} else {
				continue
			}
			if _, err = run(workspace(), os.Environ(), "gh", args...); err != nil {
				return err
			}
		}
		return nil
	}
	return fmt.Errorf("unknown Go CI operation %q", pos[0])
}
