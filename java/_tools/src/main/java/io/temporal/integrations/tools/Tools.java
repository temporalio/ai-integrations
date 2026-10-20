package io.temporal.integrations.tools;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.tomlj.Toml;
import org.tomlj.TomlArray;
import org.tomlj.TomlParseResult;

/** Infrastructure for Java-owned CI and Maven release operations. */
public final class Tools {
  static final ObjectMapper JSON = new ObjectMapper();
  static final HttpClient HTTP =
      HttpClient.newBuilder()
          .connectTimeout(Duration.ofSeconds(30))
          .followRedirects(HttpClient.Redirect.NEVER)
          .build();

  static Path root() {
    return Path.of(
            System.getenv()
                .getOrDefault(
                    "BUILD_WORKSPACE_DIRECTORY",
                    System.getenv().getOrDefault("GITHUB_WORKSPACE", ".")))
        .toAbsolutePath();
  }

  static String repository() {
    return System.getenv().getOrDefault("GITHUB_REPOSITORY", "temporalio/ai-integrations");
  }

  static List<String> strings(TomlArray values) {
    var result = new ArrayList<String>();
    if (values != null) for (int i = 0; i < values.size(); i++) result.add(values.getString(i));
    return result;
  }

  record Meta(
      String name,
      String language,
      String coordinate,
      String registry,
      String rootApi,
      String maturity,
      List<String> runtimes,
      List<String> boots,
      List<String> imports,
      List<String> smokeDependencies) {}

  static Meta meta(String directory) throws IOException {
    TomlParseResult t = Toml.parse(root().resolve(directory).resolve("plugin.toml"));
    if (t.hasErrors()) throw new IOException("Invalid plugin metadata: " + t.errors());
    return new Meta(
        t.getString("plugin.name"),
        t.getString("plugin.language"),
        t.getString("plugin.coordinate"),
        t.getString("plugin.registry"),
        t.getString("plugin.root-api"),
        t.getString("plugin.maturity"),
        strings(t.getArray("ci.runtime-versions")),
        strings(t.getArray("ci.spring-boot-versions")),
        strings(t.getArray("smoke.imports")),
        strings(t.getArray("smoke.dependencies")));
  }

  record Args(List<String> positional, Map<String, String> options) {
    String get(String key) {
      return options.get(key);
    }

    String get(String key, String fallback) {
      return options.getOrDefault(key, fallback);
    }

    String require(String key) {
      String value = get(key);
      if (value == null) throw new IllegalArgumentException("Missing --" + key);
      return value;
    }

    boolean has(String key) {
      return "true".equals(get(key));
    }
  }

  static Args arguments(String[] input) {
    var values = new LinkedHashMap<String, String>();
    var positional = new ArrayList<String>();
    for (int i = 0; i < input.length; i++) {
      String item = input[i];
      if (!item.startsWith("--")) {
        positional.add(item);
        continue;
      }
      String key = item.substring(2);
      int equals = key.indexOf('=');
      if (equals >= 0) {
        values.put(key.substring(0, equals), key.substring(equals + 1));
        continue;
      }
      if (List.of("dry-run", "require-present", "recover-only", "restore-state", "latest", "smoke")
          .contains(key)) {
        values.put(key, "true");
        continue;
      }
      if (i + 1 >= input.length || input[i + 1].startsWith("--"))
        throw new IllegalArgumentException("Missing value for " + item);
      values.put(key, input[++i]);
    }
    for (String key : List.of("dist", "state-dir", "output", "registry-json")) {
      if (values.containsKey(key)) values.put(key, root().resolve(values.get(key)).toString());
    }
    return new Args(positional, values);
  }

  static byte[] execute(
      Path directory, Map<String, String> environment, byte[] stdin, String... command)
      throws Exception {
    ProcessBuilder builder = new ProcessBuilder(command).directory(directory.toFile());
    builder.environment().putAll(environment);
    builder.redirectError(ProcessBuilder.Redirect.INHERIT);
    Process p = builder.start();
    if (stdin != null) p.getOutputStream().write(stdin);
    p.getOutputStream().close();
    byte[] result = p.getInputStream().readAllBytes();
    if (p.waitFor() != 0) throw new IOException("Command failed: " + command[0]);
    return result;
  }

  static String command(String... args) throws Exception {
    return new String(execute(root(), Map.of(), null, args), StandardCharsets.UTF_8)
        .stripTrailing();
  }

  static JsonNode gh(String endpoint, String... args) throws Exception {
    var command = new ArrayList<String>(List.of("gh", "api", endpoint));
    command.addAll(List.of(args));
    return JSON.readTree(execute(root(), Map.of(), null, command.toArray(String[]::new)));
  }

  static void outputs(Map<String, String> values, String path) throws IOException {
    StringBuilder text = new StringBuilder();
    for (var entry : values.entrySet()) {
      if (entry.getValue().contains("\n") || entry.getValue().contains("\r"))
        throw new IllegalArgumentException("GitHub output contains a line break");
      text.append(entry.getKey()).append('=').append(entry.getValue()).append('\n');
    }
    System.out.print(text);
    if (path == null) path = System.getenv("GITHUB_OUTPUT");
    if (path != null)
      Files.writeString(
          Path.of(path),
          text,
          java.nio.file.StandardOpenOption.CREATE,
          java.nio.file.StandardOpenOption.APPEND);
  }

  static byte[] get(String address, boolean missing) throws Exception {
    HttpResponse<byte[]> response =
        HTTP.send(
            HttpRequest.newBuilder(URI.create(address))
                .timeout(Duration.ofSeconds(30))
                .GET()
                .build(),
            HttpResponse.BodyHandlers.ofByteArray());
    if (missing && response.statusCode() == 404) return null;
    if (response.statusCode() != 200)
      throw new IOException(
          "Registry returned HTTP " + response.statusCode() + "; refusing to guess");
    return response.body();
  }

  @FunctionalInterface
  interface Operation {
    void run(Args args) throws Exception;
  }

  static void main(String[] args, Operation operation) {
    try {
      operation.run(arguments(args));
    } catch (Exception e) {
      System.err.println(e.getMessage());
      System.exit(1);
    }
  }

  private Tools() {}
}
