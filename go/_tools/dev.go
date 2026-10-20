package main

import (
	"errors"
	"os"
	"path/filepath"
)

func devMain(pos []string, o options) error {
	if len(pos) != 1 || pos[0] != "update" {
		return errors.New("expected update --plugin-dir go/<plugin>")
	}
	dir := filepath.Join(workspace(), o["plugin-dir"])
	for _, args := range [][]string{{"get", "-u", "-t", "./..."}, {"mod", "tidy"}, {"mod", "verify"}} {
		if _, err := run(dir, append(os.Environ(), "GOTOOLCHAIN=local"), "go", args...); err != nil {
			return err
		}
	}
	return nil
}

func lintMain(pos []string, o options) error {
	_, err := run(filepath.Join(workspace(), o["plugin-dir"]), append(os.Environ(), "GOTOOLCHAIN=local"), "go", "vet", "-mod=readonly", "./...")
	return err
}
