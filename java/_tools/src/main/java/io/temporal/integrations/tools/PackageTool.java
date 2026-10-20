package io.temporal.integrations.tools;

import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.jar.*;
import javax.tools.ToolProvider;

/** Native JAR publication, metadata validation, and isolated Maven consumer. */
public final class PackageTool {
  static void delete(Path path) throws IOException {
    if (!Files.exists(path)) return;
    Files.walkFileTree(
        path,
        new SimpleFileVisitor<Path>() {
          @Override
          public FileVisitResult visitFile(
              Path file, java.nio.file.attribute.BasicFileAttributes attrs) throws IOException {
            Files.deleteIfExists(file);
            return FileVisitResult.CONTINUE;
          }

          @Override
          public FileVisitResult visitFileFailed(Path file, IOException error) throws IOException {
            // GPG agents can remove their sockets while cleanup walks the home.
            if (error instanceof NoSuchFileException) return FileVisitResult.CONTINUE;
            throw error;
          }

          @Override
          public FileVisitResult postVisitDirectory(Path directory, IOException error)
              throws IOException {
            if (error != null && !(error instanceof NoSuchFileException)) throw error;
            Files.deleteIfExists(directory);
            return FileVisitResult.CONTINUE;
          }
        });
  }

  static void jar(Path output, Map<String, byte[]> entries) throws IOException {
    Files.createDirectories(output.toAbsolutePath().getParent());
    try (var out = new JarOutputStream(Files.newOutputStream(output))) {
      for (var e : new TreeMap<>(entries).entrySet()) {
        var entry = new JarEntry(e.getKey());
        entry.setTime(315532800000L);
        out.putNextEntry(entry);
        out.write(e.getValue());
        out.closeEntry();
      }
    }
  }

  static Map<String, byte[]> readJar(Path file) throws IOException {
    var result = new TreeMap<String, byte[]>();
    try (var zip = new java.util.zip.ZipFile(file.toFile())) {
      var entries = zip.entries();
      while (entries.hasMoreElements()) {
        var e = entries.nextElement();
        if (e.isDirectory()) continue;
        if (result.put(e.getName(), zip.getInputStream(e).readAllBytes()) != null)
          throw new IOException("duplicate JAR entry");
      }
    }
    return result;
  }

  static void build(Tools.Args args) throws Exception {
    Path project = Path.of(args.require("project")), out = Path.of(args.require("output"));
    String version = args.require("version"),
        boot = args.require("spring-boot"),
        base = "spring-ai-" + version;
    var lock =
        Tools.JSON
            .readTree(
                project
                    .resolve("maven/spring_ai_" + boot.replace('.', '_') + "_compile.json")
                    .toFile())
            .path("artifacts");
    String resolvedBoot =
        lock.path("org.springframework.boot:spring-boot-dependencies:pom")
            .path("version")
            .asText(boot);
    String resolvedAi =
        lock.path("org.springframework.ai:spring-ai-bom:pom").path("version").asText("2.0.1");
    Files.createDirectories(out);
    byte[] license = Files.readAllBytes(project.resolve("LICENSE"));
    var entries = readJar(Path.of(args.require("jar")));
    entries.put("META-INF/LICENSE", license);
    jar(out.resolve(base + ".jar"), entries);
    var source = new TreeMap<String, byte[]>();
    try (var paths = Files.walk(project.resolve("src/main/java"))) {
      for (Path p : paths.filter(Files::isRegularFile).toList())
        source.put(
            project.resolve("src/main/java").relativize(p).toString().replace('\\', '/'),
            Files.readAllBytes(p));
    }
    source.put("META-INF/LICENSE", license);
    jar(out.resolve(base + "-sources.jar"), source);
    Path docs = Files.createTempDirectory("javadoc-");
    try {
      var options =
          new ArrayList<String>(
              List.of(
                  "-quiet",
                  "-notimestamp",
                  "-Xdoclint:none",
                  "-encoding",
                  "UTF-8",
                  "-d",
                  docs.toString(),
                  "-classpath",
                  args.require("classpath")));
      try (var paths = Files.walk(project.resolve("src/main/java"))) {
        paths.filter(p -> p.toString().endsWith(".java")).forEach(p -> options.add(p.toString()));
      }
      int result =
          ToolProvider.getSystemDocumentationTool()
              .run(null, null, null, options.toArray(String[]::new));
      if (result != 0) throw new IOException("Javadoc failed");
      var docFiles = new TreeMap<String, byte[]>();
      try (var paths = Files.walk(docs)) {
        for (Path p : paths.filter(Files::isRegularFile).toList())
          docFiles.put(docs.relativize(p).toString().replace('\\', '/'), Files.readAllBytes(p));
      }
      docFiles.put("META-INF/LICENSE", license);
      jar(out.resolve(base + "-javadoc.jar"), docFiles);
    } finally {
      delete(docs);
    }
    String pom =
        Files.readString(project.resolve("publication/pom.xml"))
            .replace("<version>0.0.0</version>", "<version>" + version + "</version>")
            .replace("<version>4.0.8</version>", "<version>" + resolvedBoot + "</version>");
    pom = pom.replace("<version>2.0.1</version>", "<version>" + resolvedAi + "</version>");
    Files.writeString(out.resolve(base + ".pom"), pom);
    ObjectNode module =
        (ObjectNode) Tools.JSON.readTree(project.resolve("publication/module.json").toFile());
    module.remove("createdBy");
    ((ObjectNode) module.get("component")).put("version", version);
    for (var variant : module.withArray("variants")) {
      for (var dep : variant.path("dependencies"))
        if (dep.path("module").asText().equals("spring-boot-dependencies"))
          ((ObjectNode) dep.path("version")).put("requires", resolvedBoot);
      for (var dep : variant.path("dependencies"))
        if (dep.path("module").asText().equals("spring-ai-bom"))
          ((ObjectNode) dep.path("version")).put("requires", resolvedAi);
      for (var file : variant.path("files")) {
        var obj = (ObjectNode) file;
        String name = obj.path("name").asText().replace("0.0.0", version);
        byte[] bytes = Files.readAllBytes(out.resolve(name));
        obj.put("name", name).put("url", name).put("size", bytes.length);
        for (String algorithm : List.of("SHA-512", "SHA-256", "SHA-1", "MD5"))
          obj.put(algorithm.toLowerCase().replace("-", ""), Maven.digest(bytes, algorithm));
      }
    }
    Tools.JSON
        .writerWithDefaultPrettyPrinter()
        .writeValue(out.resolve(base + ".module").toFile(), module);
  }

  static String element(org.w3c.dom.Document doc, String name) {
    var list = doc.getDocumentElement().getElementsByTagNameNS("*", name);
    return list.getLength() == 0 ? "" : list.item(0).getTextContent();
  }

  static void check(String dir, String version, Path dist) throws Exception {
    var meta = Tools.meta(dir);
    String[] coordinate = meta.coordinate().split(":");
    String base = coordinate[1] + "-" + version;
    var expected = new TreeSet<String>();
    for (String suffix : List.of(".jar", "-sources.jar", "-javadoc.jar", ".pom", ".module"))
      expected.add(base + suffix);
    try (var files = Files.list(dist)) {
      if (!new TreeSet<>(
              files.filter(Files::isRegularFile).map(p -> p.getFileName().toString()).toList())
          .equals(expected))
        throw new IOException("Publication must contain exactly five tested artifacts");
    }
    var pom = Release.xml(Files.readAllBytes(dist.resolve(base + ".pom")));
    for (var e :
        Map.of("groupId", coordinate[0], "artifactId", coordinate[1], "version", version)
            .entrySet())
      if (!element(pom, e.getKey()).equals(e.getValue()))
        throw new IOException("POM coordinate differs");
    if (!element(pom, "name").equals("MIT") && !element(pom, "license").contains("MIT"))
      throw new IOException("POM license differs");
    var module = Tools.JSON.readTree(dist.resolve(base + ".module").toFile()).path("component");
    if (!module.path("group").asText().equals(coordinate[0])
        || !module.path("module").asText().equals(coordinate[1])
        || !module.path("version").asText().equals(version))
      throw new IOException("Module coordinate differs");
    byte[] license = Files.readAllBytes(Tools.root().resolve("LICENSE"));
    for (String file : expected)
      if (file.endsWith(".jar")) {
        var entries = readJar(dist.resolve(file));
        if (!Arrays.equals(entries.get("META-INF/LICENSE"), license))
          throw new IOException("Packaged LICENSE differs");
        for (var e : entries.entrySet())
          if (e.getKey().endsWith(".class")) {
            var input = new DataInputStream(new ByteArrayInputStream(e.getValue()));
            if (input.readInt() != 0xCAFEBABE) throw new IOException("Invalid class");
            input.readUnsignedShort();
            if (input.readUnsignedShort() > 61) throw new IOException("Class must target Java 17");
          }
      }
    var main = readJar(dist.resolve(base + ".jar"));
    for (String name : meta.imports())
      if (!main.containsKey(name.replace('.', '/') + ".class"))
        throw new IOException("Missing public class " + name);
  }

  static org.w3c.dom.Element child(org.w3c.dom.Element parent, String name) {
    for (var node = parent.getFirstChild(); node != null; node = node.getNextSibling()) {
      if (node instanceof org.w3c.dom.Element element && name.equals(element.getLocalName()))
        return element;
    }
    return null;
  }

  static String directText(org.w3c.dom.Element parent, String name) {
    var element = child(parent, name);
    return element == null ? "" : element.getTextContent();
  }

  static void smoke(String dir, String version, String repository, String token) throws Exception {
    var meta = Tools.meta(dir);
    String[] coord = meta.coordinate().split(":");
    Path temp = Files.createTempDirectory("java-consumer-");
    try {
      if (repository == null) {
        Path repo = temp.resolve("repository"),
            location = repo.resolve(coord[0].replace('.', '/') + "/" + coord[1] + "/" + version);
        Files.createDirectories(location);
        Path localDist = Tools.root().resolve("bazel-bin").resolve(dir).resolve("dist");
        if (!Files.isDirectory(localDist)) localDist = Tools.root().resolve(dir).resolve("dist");
        try (var files = Files.list(localDist)) {
          for (Path file : files.toList()) Files.copy(file, location.resolve(file.getFileName()));
        }
        repository = repo.toUri().toString();
      }
      String quoted = Tools.JSON.writeValueAsString(repository),
          include =
              "includeModule("
                  + Tools.JSON.writeValueAsString(coord[0])
                  + ","
                  + Tools.JSON.writeValueAsString(coord[1])
                  + ")",
          exclude =
              "excludeModule("
                  + Tools.JSON.writeValueAsString(coord[0])
                  + ","
                  + Tools.JSON.writeValueAsString(coord[1])
                  + ")";
      String dependency = meta.coordinate() + ":" + version;
      StringBuilder deps =
          new StringBuilder("implementation " + Tools.JSON.writeValueAsString(dependency) + "\n");
      for (String d : meta.smokeDependencies())
        deps.append("implementation ").append(Tools.JSON.writeValueAsString(d)).append('\n');
      Files.writeString(temp.resolve("settings.gradle"), "rootProject.name='isolated-consumer'\n");
      Files.writeString(
          temp.resolve("build.gradle"),
          "plugins { id 'application' }\nrepositories { maven { url=uri("
              + quoted
              + "); content { "
              + include
              + " }; if(System.getenv('CENTRAL_BEARER')) { credentials(HttpHeaderCredentials) {"
              + " name='Authorization'; value='Bearer '+System.getenv('CENTRAL_BEARER') };"
              + " authentication { header(HttpHeaderAuthentication) } } }; mavenCentral { content {"
              + " "
              + exclude
              + " } } }\ndependencies { "
              + deps
              + " }\napplication { mainClass='Consumer' }\n");
      Path source = temp.resolve("src/main/java");
      Files.createDirectories(source);
      Files.writeString(
          source.resolve("Consumer.java"),
          "public class Consumer { public static void main(String[] args) throws Exception {"
              + " for(String name: new String[] "
              + Tools.JSON.writeValueAsString(meta.imports()).replace('[', '{').replace(']', '}')
              + "){Class<?>"
              + " type=Class.forName(name);if(!type.getProtectionDomain().getCodeSource().getLocation().toString().endsWith("
              + Tools.JSON.writeValueAsString(coord[1] + "-" + version + ".jar")
              + "))throw new AssertionError(\"Unexpected plugin origin\");} } }");
      String gradle =
          com.google.devtools.build.runfiles.Runfiles.create()
              .rlocation(System.getenv("GRADLE_RUNFILE"));
      var environment = new HashMap<String, String>();
      environment.put("JAVA_HOME", System.getProperty("java.home"));
      if (token != null) environment.put("CENTRAL_BEARER", token);
      var command = new ArrayList<String>();
      if (System.getProperty("os.name").startsWith("Windows"))
        command.addAll(List.of("cmd.exe", "/c"));
      command.addAll(List.of(gradle, "--no-daemon", "run"));
      Tools.execute(temp, environment, null, command.toArray(String[]::new));
    } finally {
      delete(temp);
    }
  }

  static void run(Tools.Args args) throws Exception {
    String action = args.positional().get(0);
    if (action.equals("build")) {
      build(args);
      return;
    }
    String dir = args.require("plugin-dir"), version = args.require("version");
    if (action.equals("check")) {
      Path dist = Tools.root().resolve("bazel-bin").resolve(dir).resolve("dist");
      if (!Files.isDirectory(dist)) dist = Tools.root().resolve(dir).resolve("dist");
      check(dir, version, args.get("dist") == null ? dist : Path.of(args.get("dist")));
    } else if (action.equals("smoke"))
      smoke(dir, version, args.get("repository"), System.getenv("CENTRAL_BEARER"));
    else throw new IllegalArgumentException("Unknown Java package operation");
  }

  public static void main(String[] args) {
    Tools.main(args, PackageTool::run);
  }
}
