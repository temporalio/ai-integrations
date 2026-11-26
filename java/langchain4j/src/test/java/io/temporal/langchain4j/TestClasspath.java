package io.temporal.langchain4j;

import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/** Launches test subprocesses with the test runtime classpath, which Gradle writes to a file. */
final class TestClasspath {

  private TestClasspath() {}

  /** Returns the test runtime classpath entries. */
  static String[] entries() {
    String classpathFile = System.getProperty("langchain4j.test.classpath-file");
    assertNotNull(classpathFile, "run through Gradle; see build.gradle");
    try {
      String classpath = Files.readString(Path.of(classpathFile), StandardCharsets.UTF_8);
      return classpath.split(Pattern.quote(File.pathSeparator));
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  /**
   * Writes a {@code java} argument file that runs {@code mainClass} with the classpath entries that
   * pass {@code keep}; an argument file keeps long classpaths under the Windows command-line limit.
   */
  static Path argumentFile(Path directory, Class<?> mainClass, Predicate<String> keep) {
    String classpath =
        Arrays.stream(entries()).filter(keep).collect(Collectors.joining(File.pathSeparator));
    try {
      Path arguments = Files.createTempFile(directory, mainClass.getSimpleName(), ".args");
      Files.writeString(
          arguments,
          "-cp\n\""
              + classpath.replace("\\", "\\\\").replace("\"", "\\\"")
              + "\"\n"
              + mainClass.getName()
              + "\n",
          StandardCharsets.UTF_8);
      return arguments;
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  /** Returns the {@code java} executable of the running JVM. */
  static String java() {
    return Path.of(System.getProperty("java.home"), "bin", "java").toString();
  }
}
