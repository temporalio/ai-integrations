package io.temporal.integrations.tools;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.UUID;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

/** Sign, privately stage, recover, and publish exactly the distributions tested by CI. */
public final class Maven {
  static final String PORTAL = "https://central.sonatype.com/api/v1/publisher";
  static final List<String> HASHES = List.of("MD5", "SHA-1", "SHA-256");
  static final List<String> SUFFIXES =
      List.of(".jar", "-sources.jar", "-javadoc.jar", ".pom", ".module");

  static String digest(byte[] bytes, String algorithm) throws Exception {
    return HexFormat.of().formatHex(MessageDigest.getInstance(algorithm).digest(bytes));
  }

  static String checksumSuffix(String algorithm) {
    return algorithm.replace("-", "").toLowerCase();
  }

  static Map<String, byte[]> artifacts(Path dist, String coordinate, String version)
      throws Exception {
    Release.Version.parse(version);
    if (!coordinate.matches("[a-zA-Z0-9_.-]+:[a-zA-Z0-9_.-]+"))
      throw new IllegalArgumentException("Invalid Maven coordinate");
    String[] parts = coordinate.split(":");
    var expected = new HashSet<String>();
    for (String suffix : SUFFIXES) expected.add(parts[1] + "-" + version + suffix);
    var contents = new TreeMap<String, byte[]>();
    try (var files = Files.list(dist)) {
      for (Path file : files.toList()) {
        if (!expected.remove(file.getFileName().toString())
            || !Files.isRegularFile(file)
            || Files.isSymbolicLink(file))
          throw new IllegalArgumentException("Expected exactly five regular Maven artifacts");
        contents.put(
            parts[0].replace('.', '/') + "/" + parts[1] + "/" + version + "/" + file.getFileName(),
            Files.readAllBytes(file));
      }
    }
    if (!expected.isEmpty()) throw new IllegalArgumentException("Missing Maven artifacts");
    return contents;
  }

  static java.util.Set<String> names(Map<String, byte[]> raw) {
    var names = new HashSet<String>();
    for (String name : raw.keySet()) {
      names.add(name);
      names.add(name + ".asc");
      for (String algorithm : HASHES) {
        names.add(name + "." + checksumSuffix(algorithm));
        names.add(name + ".asc." + checksumSuffix(algorithm));
      }
    }
    return names;
  }

  static Map<String, byte[]> validateBundle(Path bundle, Map<String, byte[]> raw) throws Exception {
    var files = new TreeMap<String, byte[]>();
    try (var zip = new ZipFile(bundle.toFile())) {
      var entries = zip.entries();
      while (entries.hasMoreElements()) {
        ZipEntry entry = entries.nextElement();
        if (files.containsKey(entry.getName()))
          throw new IllegalArgumentException("Duplicate signed bundle member");
        files.put(entry.getName(), zip.getInputStream(entry).readAllBytes());
      }
    }
    if (!files.keySet().equals(names(raw)))
      throw new IllegalArgumentException("Signed bundle file set differs");
    for (var entry : raw.entrySet()) {
      String name = entry.getKey();
      if (!Arrays.equals(files.get(name), entry.getValue()))
        throw new IllegalArgumentException("Saved bundle differs from tested bytes");
      if (!new String(files.get(name + ".asc"), StandardCharsets.US_ASCII)
          .startsWith("-----BEGIN PGP SIGNATURE-----"))
        throw new IllegalArgumentException("Missing armored signature");
      for (String signed : List.of(name, name + ".asc"))
        for (String algorithm : HASHES)
          if (!new String(
                  files.get(signed + "." + checksumSuffix(algorithm)), StandardCharsets.US_ASCII)
              .strip()
              .equals(digest(files.get(signed), algorithm)))
            throw new IllegalArgumentException("Incorrect signed bundle checksum");
    }
    return files;
  }

  static void writeBundle(Path path, Map<String, byte[]> contents) throws Exception {
    Files.createDirectories(path.toAbsolutePath().getParent());
    try (var zip = new ZipOutputStream(Files.newOutputStream(path))) {
      for (var entry : new TreeMap<>(contents).entrySet()) {
        var member = new ZipEntry(entry.getKey());
        member.setTime(315532800000L);
        zip.putNextEntry(member);
        zip.write(entry.getValue());
        zip.closeEntry();
      }
    }
  }

  static byte[] keyPayload(String value) {
    if (value.stripLeading().startsWith("-----BEGIN PGP PRIVATE KEY BLOCK-----"))
      return value.getBytes(StandardCharsets.UTF_8);
    try {
      return Base64.getDecoder().decode(value.replaceAll("\\s", ""));
    } catch (IllegalArgumentException e) {
      throw new IllegalArgumentException(
          "GPG_PRIVATE_KEY must be armored or a base64 secret keyring");
    }
  }

  static byte[] gpg(Path home, byte[] input, String... arguments) throws Exception {
    var command =
        new ArrayList<String>(
            List.of(
                "gpg", "--homedir", home.toString(), "--batch", "--no-tty", "--status-fd", "2"));
    command.addAll(List.of(arguments));
    Process process = new ProcessBuilder(command).start();
    var diagnostics = new ByteArrayOutputStream();
    Thread reader =
        new Thread(
            () -> {
              try {
                process.getErrorStream().transferTo(diagnostics);
              } catch (IOException ignored) {
              }
            });
    reader.start();
    if (input != null) process.getOutputStream().write(input);
    process.getOutputStream().close();
    byte[] output = process.getInputStream().readAllBytes();
    int status = process.waitFor();
    reader.join();
    if (status != 0) {
      var allowed =
          java.util.regex.Pattern.compile(
                  "(?m)^\\[GNUPG:\\]"
                      + " (BAD_PASSPHRASE|MISSING_PASSPHRASE|NODATA|BADARMOR|KEYEXPIRED|KEYREVOKED|INV_SGNR|NO_SECKEY|NO_PUBKEY|FAILURE|ERROR)(?:"
                      + " |$)")
              .matcher(diagnostics.toString(StandardCharsets.UTF_8));
      var codes = new java.util.TreeSet<String>();
      while (allowed.find()) codes.add(allowed.group(1));
      throw new IOException(
          "GPG operation failed (exit "
              + status
              + "; "
              + codes
              + "); verify signing configuration");
    }
    return output;
  }

  static void signBundle(Path bundle, Map<String, byte[]> raw) throws Exception {
    String key = System.getenv("GPG_PRIVATE_KEY"),
        passphrase = System.getenv("GPG_PASSPHRASE"),
        fingerprint = System.getenv().getOrDefault("GPG_FINGERPRINT", "").toUpperCase();
    if (key == null || passphrase == null || !fingerprint.matches("(?:[0-9A-F]{40}|[0-9A-F]{64})"))
      throw new IllegalArgumentException(
          "Set private key, passphrase, and full signing fingerprint");
    // GPG's Unix sockets must fit macOS's short socket-path limit.
    Path home = Files.createTempDirectory("gpg-");
    home.toFile().setReadable(false, false);
    home.toFile().setReadable(true, true);
    home.toFile().setWritable(false, false);
    home.toFile().setWritable(true, true);
    home.toFile().setExecutable(false, false);
    home.toFile().setExecutable(true, true);
    try {
      gpg(home, keyPayload(key), "--import");
      String listing =
          new String(
              gpg(home, null, "--with-colons", "--list-secret-keys"), StandardCharsets.UTF_8);
      boolean found = false;
      for (String line : listing.split("\n"))
        if (line.startsWith("fpr:") && line.split(":", -1)[9].equals(fingerprint)) found = true;
      if (!found)
        throw new IllegalArgumentException("Signing fingerprint differs from imported key");
      var contents = new TreeMap<String, byte[]>();
      for (var entry : raw.entrySet()) {
        Path file = home.resolve(Path.of(entry.getKey()).getFileName()),
            signature = Path.of(file + ".asc");
        Files.write(file, entry.getValue());
        gpg(
            home,
            (passphrase + "\n").getBytes(StandardCharsets.UTF_8),
            "--pinentry-mode",
            "loopback",
            "--passphrase-fd",
            "0",
            "--local-user",
            fingerprint,
            "--armor",
            "--detach-sign",
            "--output",
            signature.toString(),
            file.toString());
        gpg(home, null, "--verify", signature.toString(), file.toString());
        contents.put(entry.getKey(), entry.getValue());
        contents.put(entry.getKey() + ".asc", Files.readAllBytes(signature));
        for (String signed : List.of(entry.getKey(), entry.getKey() + ".asc"))
          for (String algorithm : HASHES)
            contents.put(
                signed + "." + checksumSuffix(algorithm),
                digest(contents.get(signed), algorithm).getBytes(StandardCharsets.US_ASCII));
      }
      writeBundle(bundle, contents);
    } finally {
      try {
        new ProcessBuilder("gpgconf", "--homedir", home.toString(), "--kill", "gpg-agent")
            .start()
            .waitFor();
      } catch (Exception ignored) {
      }
      PackageTool.delete(home);
    }
  }

  static String deploymentId(String value) {
    String parsed = UUID.fromString(value).toString();
    if (!parsed.equals(value))
      throw new IllegalArgumentException("Deployment ID must be a canonical UUID");
    return parsed;
  }

  static final class Portal {
    final String token;

    Portal() {
      String user = System.getenv("CENTRAL_USERNAME"), password = System.getenv("CENTRAL_PASSWORD");
      if (user == null || password == null)
        throw new IllegalArgumentException("Central user-token credentials are required");
      token =
          Base64.getEncoder()
              .encodeToString((user + ":" + password).getBytes(StandardCharsets.UTF_8));
      if ("true".equals(System.getenv("GITHUB_ACTIONS")))
        System.out.println("::add-mask::" + token);
    }

    byte[] request(String path, String method, byte[] body, String contentType, boolean absent)
        throws Exception {
      URI address = URI.create(PORTAL + path);
      String authorization = "Bearer " + token;
      for (int redirects = 0; redirects <= 5; redirects++) {
        var builder = HttpRequest.newBuilder(address).timeout(Duration.ofSeconds(60));
        if (authorization != null) builder.header("Authorization", authorization);
        if (contentType != null) builder.header("Content-Type", contentType);
        builder.method(
            method,
            body == null
                ? HttpRequest.BodyPublishers.noBody()
                : HttpRequest.BodyPublishers.ofByteArray(body));
        HttpResponse<byte[]> response;
        try {
          response = Tools.HTTP.send(builder.build(), HttpResponse.BodyHandlers.ofByteArray());
        } catch (Exception e) {
          throw new IOException(
              "Central " + method + " failed; inspect Portal before retrying an upload");
        }
        if (response.statusCode() == 404 && absent) return null;
        if (response.statusCode() >= 300 && response.statusCode() < 400) {
          URI next = address.resolve(response.headers().firstValue("location").orElseThrow());
          if (!next.getScheme().equals("https"))
            throw new IOException("Refusing non-HTTPS registry redirect");
          if (!next.getAuthority().equals(address.getAuthority())) authorization = null;
          address = next;
          if (response.statusCode() == 303) {
            method = "GET";
            body = null;
          }
          continue;
        }
        if (response.statusCode() < 200 || response.statusCode() >= 300)
          throw new IOException("Central " + method + " returned HTTP " + response.statusCode());
        return response.body();
      }
      throw new IOException("Too many Portal redirects");
    }

    JsonNode status(String deployment) throws Exception {
      JsonNode result =
          Tools.JSON.readTree(request("/status?id=" + deployment, "POST", null, null, false));
      if (!result.path("deploymentId").asText().equals(deployment)
          || !result.has("deploymentState")) throw new IOException("Invalid deployment status");
      return result;
    }

    String waitFor(String deployment, boolean published, long deadline) throws Exception {
      while (true) {
        String state = status(deployment).path("deploymentState").asText();
        System.out.println("Central deployment " + deployment + ": " + state);
        if (state.equals("PUBLISHED") || state.equals("VALIDATED") && !published) return state;
        if (!List.of("PENDING", "VALIDATING", "VALIDATED", "PUBLISHING").contains(state))
          throw new IOException(
              "Central validation failed or returned unknown state; inspect Portal");
        long remaining = deadline - System.nanoTime();
        if (remaining <= 0)
          throw new IOException("Central polling timed out; recover using saved deployment state");
        Thread.sleep(Math.min(10000, Math.max(1, remaining / 1000000)));
      }
    }

    byte[] download(String deployment, String name) throws Exception {
      return request("/deployment/" + deployment + "/download/" + name, "GET", null, null, false);
    }

    String upload(Path bundle, String coordinate, String version) throws Exception {
      String boundary = "central-" + UUID.randomUUID().toString().replace("-", "");
      var body = new ByteArrayOutputStream();
      body.write(
          ("--"
                  + boundary
                  + "\r\n"
                  + "Content-Disposition: form-data; name=\"bundle\"; filename=\"bundle.zip\"\r\n"
                  + "Content-Type: application/octet-stream\r\n\r\n")
              .getBytes(StandardCharsets.UTF_8));
      body.write(Files.readAllBytes(bundle));
      body.write(("\r\n--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8));
      return deploymentId(
          new String(
                  request(
                      "/upload?name="
                          + URLEncoder.encode(coordinate + ":" + version, StandardCharsets.UTF_8)
                          + "&publishingType=USER_MANAGED",
                      "POST",
                      body.toByteArray(),
                      "multipart/form-data; boundary=" + boundary,
                      false),
                  StandardCharsets.UTF_8)
              .strip());
    }
  }

  static Map<String, String> hashes(Map<String, byte[]> raw) throws Exception {
    var result = new TreeMap<String, String>();
    for (var entry : raw.entrySet())
      result.put(entry.getKey(), digest(entry.getValue(), "SHA-256"));
    return result;
  }

  static JsonNode saveState(
      Path directory, String deployment, String coordinate, String version, Map<String, byte[]> raw)
      throws Exception {
    var state = Tools.JSON.createObjectNode();
    state.put("deployment_id", deployment);
    state.put("coordinate", coordinate);
    state.put("version", version);
    state.set("artifacts", Tools.JSON.valueToTree(hashes(raw)));
    state.put(
        "bundle_sha256", digest(Files.readAllBytes(directory.resolve("bundle.zip")), "SHA-256"));
    Path temp = directory.resolve("deployment.json.tmp");
    Tools.JSON.writerWithDefaultPrettyPrinter().writeValue(temp.toFile(), state);
    Files.move(
        temp,
        directory.resolve("deployment.json"),
        StandardCopyOption.REPLACE_EXISTING,
        StandardCopyOption.ATOMIC_MOVE);
    return state;
  }

  static JsonNode loadState(
      Path directory, String coordinate, String version, Map<String, byte[]> raw) throws Exception {
    Path file = directory.resolve("deployment.json");
    if (!Files.exists(file)) return null;
    JsonNode state = Tools.JSON.readTree(file.toFile());
    if (!state.path("coordinate").asText().equals(coordinate)
        || !state.path("version").asText().equals(version)
        || !state.path("artifacts").equals(Tools.JSON.valueToTree(hashes(raw))))
      throw new IllegalArgumentException("Saved deployment differs from tested artifacts");
    deploymentId(state.path("deployment_id").asText());
    Path bundle = directory.resolve("bundle.zip");
    if (!Files.isRegularFile(bundle)
        || !digest(Files.readAllBytes(bundle), "SHA-256")
            .equals(state.path("bundle_sha256").asText()))
      throw new IllegalArgumentException("Saved bundle is missing or changed");
    validateBundle(bundle, raw);
    return state;
  }

  @FunctionalInterface
  interface Fetch {
    byte[] get(String name) throws Exception;
  }

  static void verifyRemote(Map<String, byte[]> files, Fetch fetch, int attempts) throws Exception {
    for (var entry : files.entrySet()) {
      for (int attempt = 0; attempt < attempts; attempt++) {
        byte[] remote = fetch.get(entry.getKey());
        if (remote != null) {
          if (!Arrays.equals(remote, entry.getValue()))
            throw new IOException("Registry bytes differ; uploaded artifacts are immutable");
          break;
        }
        if (attempt + 1 == attempts)
          throw new IOException("Registry does not serve " + entry.getKey());
        Thread.sleep(10000);
      }
    }
  }

  static void verifyDeployment(Portal portal, String deployment, String coordinate, String version)
      throws Exception {
    String[] parts = coordinate.split(":");
    var expected =
        Tools.JSON.valueToTree(List.of("pkg:maven/" + parts[0] + "/" + parts[1] + "@" + version));
    if (!portal.status(deployment).path("purls").equals(expected))
      throw new IOException("Deployment includes a different coordinate or version");
  }

  static void requestedDeployment(String requested, String saved) {
    if (requested != null && !requested.isEmpty() && !requested.equals(saved))
      throw new IllegalArgumentException("Requested deployment differs from saved state");
  }

  static void run(Tools.Args args) throws Exception {
    String action = args.positional().get(0),
        dir = args.require("plugin-dir"),
        version = args.require("version");
    var meta = Tools.meta(dir);
    if (!meta.language().equals("java") || !meta.registry().equals("maven"))
      throw new IllegalArgumentException("Expected Java Maven plugin");
    Path dist = Path.of(args.require("dist")),
        stateDir = Tools.root().resolve(args.get("state-dir", "release-state"));
    Files.createDirectories(stateDir);
    var raw = artifacts(dist, meta.coordinate(), version);
    PackageTool.check(dir, version, dist);
    Path bundle = stateDir.resolve("bundle.zip");
    if (action.equals("bundle")) {
      signBundle(bundle, raw);
      validateBundle(bundle, raw);
      return;
    }
    Portal portal = new Portal();
    JsonNode state = loadState(stateDir, meta.coordinate(), version, raw);
    long deadline = System.nanoTime() + Duration.ofMinutes(30).toNanos();
    String deployment;
    if (action.equals("stage")) {
      if (state != null) {
        deployment = state.path("deployment_id").asText();
        requestedDeployment(args.get("deployment-id"), deployment);
      } else if (args.get("deployment-id") != null && !args.get("deployment-id").isEmpty()) {
        deployment = deploymentId(args.get("deployment-id"));
        boolean published = portal.waitFor(deployment, false, deadline).equals("PUBLISHED");
        var contents = new TreeMap<String, byte[]>();
        for (String name : names(raw)) {
          byte[] value =
              published
                  ? Tools.get(Release.CENTRAL + "/" + name, false)
                  : portal.download(deployment, name);
          contents.put(name, value);
        }
        writeBundle(bundle, contents);
        validateBundle(bundle, raw);
        state = saveState(stateDir, deployment, meta.coordinate(), version, raw);
      } else {
        if (args.has("recover-only") || Files.exists(stateDir.resolve("upload-started.json")))
          throw new IllegalArgumentException(
              "Prior upload may have succeeded; recovery requires saved state or an explicit"
                  + " deployment ID");
        String primary =
            raw.keySet().stream()
                .filter(name -> name.endsWith("/" + meta.name() + "-" + version + ".jar"))
                .findFirst()
                .orElseThrow();
        if (Tools.get(Release.CENTRAL + "/" + primary, true) != null
            || portal.request("/deployments/download/" + primary, "GET", null, null, true) != null)
          throw new IllegalArgumentException(
              "Version already exists; recover using its deployment ID");
        signBundle(bundle, raw);
        validateBundle(bundle, raw);
        Tools.JSON.writeValue(
            stateDir.resolve("upload-started.json").toFile(),
            Map.of("coordinate", meta.coordinate(), "version", version));
        deployment = portal.upload(bundle, meta.coordinate(), version);
        state = saveState(stateDir, deployment, meta.coordinate(), version, raw);
      }
      Tools.outputs(Map.of("deployment_id", deployment), args.get("github-output"));
      boolean published = portal.waitFor(deployment, false, deadline).equals("PUBLISHED");
      verifyDeployment(portal, deployment, meta.coordinate(), version);
      if (published && Release.Version.parse(version).rc() != null)
        throw new IllegalArgumentException(
            "Candidate has been published; candidates must remain private");
      var contents = validateBundle(bundle, raw);
      if (published)
        verifyRemote(contents, name -> Tools.get(Release.CENTRAL + "/" + name, true), 30);
      else verifyRemote(contents, name -> portal.download(deployment, name), 1);
      if (args.has("smoke"))
        PackageTool.smoke(
            dir,
            version,
            published ? Release.CENTRAL : PORTAL + "/deployment/" + deployment + "/download/",
            published ? null : portal.token);
    } else if (action.equals("publish")) {
      if (Release.Version.parse(version).rc() != null)
        throw new IllegalArgumentException("Candidates cannot be published");
      Release.run(
          new Tools.Args(
              List.of("check-version-policy"), Map.of("plugin-dir", dir, "version", version)));
      if (state == null)
        throw new IllegalArgumentException("Publication requires saved verified staging state");
      deployment = state.path("deployment_id").asText();
      String status = portal.waitFor(deployment, false, deadline);
      verifyDeployment(portal, deployment, meta.coordinate(), version);
      if (status.equals("VALIDATED"))
        portal.request("/deployment/" + deployment, "POST", null, null, false);
      portal.waitFor(deployment, true, deadline);
      verifyRemote(
          validateBundle(bundle, raw), name -> Tools.get(Release.CENTRAL + "/" + name, true), 30);
      if (args.has("smoke")) PackageTool.smoke(dir, version, Release.CENTRAL, null);
    } else throw new IllegalArgumentException("Unknown Maven operation");
  }
}
