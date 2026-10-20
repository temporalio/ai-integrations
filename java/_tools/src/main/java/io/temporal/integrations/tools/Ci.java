package io.temporal.integrations.tools;

import java.nio.file.*;
import java.util.*;

/** Metadata-driven Java CI matrix and conventions. */
public final class Ci {
  static void run(Tools.Args args) throws Exception {
    String action = args.positional().get(0);
    if (action.equals("matrix")) {
      var meta = Tools.meta(args.require("plugin-dir"));
      var versions = meta.runtimes();
      String min = versions.get(0), max = versions.get(versions.size() - 1);
      var cells = new ArrayList<Map<String, Object>>();
      for (String boot : meta.boots()) {
        for (String runtime : new LinkedHashSet<>(List.of(min, max)))
          cells.add(
              Map.of(
                  "os",
                  "ubuntu-latest",
                  "runtime",
                  runtime,
                  "spring_boot",
                  boot,
                  "dist",
                  runtime.equals(max) && boot.equals(meta.boots().get(0))));
        for (String os : List.of("macos-latest", "windows-latest"))
          cells.add(Map.of("os", os, "runtime", max, "spring_boot", boot, "dist", false));
      }
      Tools.outputs(
          Map.of("matrix", Tools.JSON.writeValueAsString(Map.of("include", cells))),
          args.get("github-output"));
      return;
    }
    var names = new ArrayList<String>();
    try (var paths = Files.list(Tools.root().resolve("java"))) {
      for (Path p : paths.filter(Files::isDirectory).toList())
        if (!p.getFileName().toString().startsWith("_") && Files.exists(p.resolve("plugin.toml")))
          names.add(p.getFileName().toString());
    }
    Collections.sort(names);
    if (action.equals("select")) {
      var selected = new TreeSet<String>();
      String event = System.getenv("GITHUB_EVENT_NAME");
      if ("pull_request".equals(event) || args.get("base") != null) {
        try {
          String base =
              args.get("base", "origin/" + System.getenv().getOrDefault("GITHUB_BASE_REF", "main"));
          for (String file :
              Tools.command("git", "diff", "--name-only", "--no-renames", base + "...HEAD")
                  .split("\n")) {
            if (file.matches(
                "^(\\.github/|bazel/|MODULE\\.bazel|BUILD\\.bazel|REPO\\.bazel|\\.bazel|LICENSE$).*")) {
              selected.addAll(names);
              break;
            }
            if (!file.startsWith("java/")) continue;
            String name = file.split("/")[1];
            if (names.contains(name)) selected.add(name);
            else selected.addAll(names);
          }
        } catch (Exception e) {
          selected.addAll(names);
        }
      } else selected.addAll(names);
      Tools.outputs(
          Map.of("java", Tools.JSON.writeValueAsString(selected)), args.get("github-output"));
    } else if (action.equals("conventions")) {
      byte[] license = Files.readAllBytes(Tools.root().resolve("LICENSE"));
      for (String name : names) {
        String dir = "java/" + name;
        var meta = Tools.meta(dir);
        if (!meta.name().equals(name)
            || !meta.language().equals("java")
            || !meta.registry().equals("maven")
            || !meta.coordinate().equals("io.temporal:" + name)
            || !Release.FIRST.containsKey(meta.maturity())
            || meta.runtimes().isEmpty()
            || meta.boots().isEmpty())
          throw new IllegalArgumentException("Invalid Java metadata " + name);
        for (String file : List.of("BUILD.bazel", "README.md", "LICENSE"))
          if (!Files.isRegularFile(Tools.root().resolve(dir).resolve(file)))
            throw new IllegalArgumentException("Missing " + file);
        if (!Arrays.equals(
            license, Files.readAllBytes(Tools.root().resolve(dir).resolve("LICENSE"))))
          throw new IllegalArgumentException("LICENSE differs");
      }
    } else throw new IllegalArgumentException("Unknown Java CI operation");
  }

  public static void main(String[] args) {
    Tools.main(args, Ci::run);
  }
}
