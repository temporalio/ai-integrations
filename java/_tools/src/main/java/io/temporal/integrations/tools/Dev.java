package io.temporal.integrations.tools;

import java.nio.file.*;
import java.util.*;
import java.util.regex.*;

/** Explicit compatible-family dependency refresh, followed by Bazel lock regeneration. */
public final class Dev {
  static void run(Tools.Args args) throws Exception {
    if (!args.positional().get(0).equals("update"))
      throw new IllegalArgumentException("Expected update --plugin-dir java/<plugin>");
    String dir = args.require("plugin-dir");
    var meta = Tools.meta(dir);
    if (!meta.language().equals("java")) throw new IllegalArgumentException("Expected Java plugin");
    Path module = Tools.root().resolve("java/_deps.MODULE.bazel");
    String text = Files.readString(module);
    var pattern =
        Pattern.compile(
            "(io\\.temporal|org\\.springframework\\.ai|org\\.springframework\\.boot):([a-zA-Z0-9_.-]+):([0-9]+\\.[0-9]+\\.[0-9]+)");
    var matcher = pattern.matcher(text);
    var replacements = new HashMap<String, String>();
    while (matcher.find()) {
      String coordinate = matcher.group(1) + ":" + matcher.group(2),
          old = matcher.group(3),
          major = old.split("\\.")[0];
      if (replacements.containsKey(matcher.group())) continue;
      String latest = old;
      for (String version : Release.published(coordinate, null))
        if (version.matches(major + "\\.[0-9]+\\.[0-9]+")
            && Release.Version.parse(version).compareTo(Release.Version.parse(latest)) > 0)
          latest = version;
      replacements.put(matcher.group(), coordinate + ":" + latest);
    }
    for (var e : replacements.entrySet()) text = text.replace(e.getKey(), e.getValue());
    Files.writeString(module, text);
    try (var paths = Files.list(Tools.root().resolve(dir).resolve("maven"))) {
      for (Path lock : paths.filter(p -> p.toString().endsWith(".json")).toList()) {
        String repo = lock.getFileName().toString().replace(".json", "");
        Tools.execute(
            Tools.root(),
            Map.of("REPIN", "1"),
            null,
            System.getenv().getOrDefault("BAZEL", "bazel"),
            "run",
            "--lockfile_mode=update",
            "@" + repo + "//:pin");
      }
    }
  }

  public static void main(String[] args) {
    Tools.main(args, Dev::run);
  }
}
