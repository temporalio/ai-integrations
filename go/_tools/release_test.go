package main

import (
	"archive/zip"
	"os"
	"path/filepath"
	"testing"
)

func TestCanonicalVersions(t *testing.T) {
	for _, v := range []string{"0.1.0", "0.1.0-rc.1", "1.24.0"} {
		if err := validVersion(v); err != nil {
			t.Fatal(err)
		}
	}
	for _, v := range []string{"v0.1.0", "01.1.0", "0.1", "0.1.0+build", "0.1.0-rc.01", "0.1.0\n"} {
		if validVersion(v) == nil {
			t.Fatalf("accepted %q", v)
		}
	}
}
func TestFirstVersionAndOrdering(t *testing.T) {
	if err := checkPolicy("0.1.0-rc.1", "public-preview", nil); err != nil {
		t.Fatal(err)
	}
	for _, v := range []string{"0.0.1", "1.0.0"} {
		if checkPolicy(v, "public-preview", nil) == nil {
			t.Fatal("accepted wrong first release")
		}
	}
	if checkPolicy("0.1.0", "public-preview", []string{"0.1.1"}) == nil {
		t.Fatal("accepted older version")
	}
	if err := checkPolicy("0.1.1", "public-preview", []string{"0.1.1"}); err != nil {
		t.Fatal(err)
	}
}
func TestTagValidation(t *testing.T) {
	for _, tag := range []string{"python/x/v0.1.0", "go/../v0.1.0", "go/x/v2.0.0", "go/x/v0.1"} {
		if _, err := parseTag(tag); err == nil {
			t.Fatal("accepted", tag)
		}
	}
	out, err := parseTag("go/googleadk/v0.1.0-rc.1")
	if err != nil || out["prerelease"] != "true" {
		t.Fatal(out, err)
	}
}
func TestVanityRequiresExactMapping(t *testing.T) {
	valid := []byte(`<meta name="go-import" content="go.temporal.io/googleadk git https://github.com/temporalio/ai-integrations.git go/googleadk">`)
	if err := verifyVanityHTML(valid, "go.temporal.io/googleadk", "temporalio/ai-integrations", "go/googleadk"); err != nil {
		t.Fatal(err)
	}
	if verifyVanityHTML(valid, "go.temporal.io/googleadk", "temporalio/ai-integrations", "go/other") == nil {
		t.Fatal("accepted wrong subdirectory")
	}
}
func TestSelection(t *testing.T) {
	names := []string{"googleadk"}
	if len(selected([]string{"README.md"}, names)) != 0 {
		t.Fatal("README selected plugin")
	}
	for _, file := range []string{"bazel/python.bzl", ".github/workflows/ci.yml", "go/googleadk/main.go", "go/_shared/config"} {
		if len(selected([]string{file}, names)) != 1 {
			t.Fatal(file)
		}
	}
}
func TestPublicArchiveMustMatchAllTestedFiles(t *testing.T) {
	tmp := t.TempDir()
	dir := filepath.Join(tmp, "go/googleadk")
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module go.temporal.io/googleadk\n"), 0600)
	if _, err := run(tmp, os.Environ(), "git", "init", "-q"); err != nil {
		t.Fatal(err)
	}
	if _, err := run(tmp, os.Environ(), "git", "add", "."); err != nil {
		t.Fatal(err)
	}
	for _, mismatch := range []bool{false, true} {
		archive := filepath.Join(tmp, "module.zip")
		f, _ := os.Create(archive)
		z := zip.NewWriter(f)
		w, _ := z.Create("go.temporal.io/googleadk@v0.1.0/go.mod")
		payload := []byte("module go.temporal.io/googleadk\n")
		if mismatch {
			payload = []byte("changed")
		}
		w.Write(payload)
		z.Close()
		f.Close()
		err := verifyZip(tmp, "go/googleadk", "go.temporal.io/googleadk", "0.1.0", archive)
		if (err != nil) != mismatch {
			t.Fatal(mismatch, err)
		}
	}
}
