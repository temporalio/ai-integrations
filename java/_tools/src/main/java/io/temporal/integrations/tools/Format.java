package io.temporal.integrations.tools;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

/** Resolve workspace filenames before delegating to the pinned formatter. */
public final class Format {
  public static void main(String[] args) throws Exception {
    Path root = Path.of(System.getenv().getOrDefault("BUILD_WORKSPACE_DIRECTORY", "."));
    var files = new ArrayList<String>();
    for (int i = 0; i < args.length; i++) {
      if (args[i].equals("--plugin-dir")) {
        Path source = root.resolve(args[++i]).resolve("src");
        try (var paths = Files.walk(source)) {
          paths
              .filter(p -> p.toString().endsWith(".java"))
              .sorted()
              .forEach(p -> files.add(p.toString()));
        }
      } else {
        files.add(
            args[i].startsWith("-") || Path.of(args[i]).isAbsolute()
                ? args[i]
                : root.resolve(args[i]).toString());
      }
    }
    com.google.googlejavaformat.java.Main.main(files.toArray(String[]::new));
  }
}
