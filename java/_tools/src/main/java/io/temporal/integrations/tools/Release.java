package io.temporal.integrations.tools;

import com.fasterxml.jackson.databind.JsonNode;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import javax.xml.parsers.DocumentBuilderFactory;

/** Maven version and publication policy, independent of the Python tooling. */
public final class Release {
  static final Pattern VERSION =
      Pattern.compile("(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)(?:-RC([1-9][0-9]*))?");
  static final Map<String, String> FIRST =
      Map.of("generally-available", "1.0.0", "public-preview", "0.1.0", "pre-release", "0.0.1");
  static final String CENTRAL = "https://repo.maven.apache.org/maven2";

  record Version(int major, int minor, int patch, Integer rc) implements Comparable<Version> {
    static Version parse(String raw) {
      var m = VERSION.matcher(raw);
      if (!m.matches())
        throw new IllegalArgumentException("Maven version must be canonical X.Y.Z[-RCN]");
      return new Version(
          Integer.parseInt(m.group(1)),
          Integer.parseInt(m.group(2)),
          Integer.parseInt(m.group(3)),
          m.group(4) == null ? null : Integer.valueOf(m.group(4)));
    }

    public int compareTo(Version other) {
      int result =
          Comparator.comparingInt(Version::major)
              .thenComparingInt(Version::minor)
              .thenComparingInt(Version::patch)
              .compare(this, other);
      if (result != 0) return result;
      if (rc == null) return other.rc == null ? 0 : 1;
      if (other.rc == null) return -1;
      return rc.compareTo(other.rc);
    }

    String base() {
      return major + "." + minor + "." + patch;
    }
  }

  static void checkPolicy(String raw, String maturity, List<String> published) {
    Version version = Version.parse(raw);
    String first = FIRST.get(maturity);
    if (first == null) throw new IllegalArgumentException("Unknown maturity");
    if (published.isEmpty() && !version.base().equals(first))
      throw new IllegalArgumentException("First release must be " + first);
    for (String old : published)
      if (version.compareTo(Version.parse(old)) < 0)
        throw new IllegalArgumentException("Version is older than published " + old);
  }

  static Map<String, String> parseTag(String tag) {
    var m = Pattern.compile("java/([a-z0-9][a-z0-9_.-]*)/v(.+)").matcher(tag);
    if (!m.matches() || m.group(1).contains(".."))
      throw new IllegalArgumentException("Expected java/<plugin>/v<version>");
    Version version = Version.parse(m.group(2));
    return new LinkedHashMap<>(
        Map.of(
            "tag",
            tag,
            "language",
            "java",
            "plugin",
            m.group(1),
            "plugin_dir",
            "java/" + m.group(1),
            "version",
            m.group(2),
            "prerelease",
            String.valueOf(version.rc() != null)));
  }

  static org.w3c.dom.Document xml(byte[] bytes) throws Exception {
    var factory = DocumentBuilderFactory.newInstance();
    factory.setNamespaceAware(true);
    factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
    factory.setFeature("http://xml.org/sax/features/external-general-entities", false);
    factory.setFeature("http://xml.org/sax/features/external-parameter-entities", false);
    return factory.newDocumentBuilder().parse(new java.io.ByteArrayInputStream(bytes));
  }

  static List<String> published(String coordinate, String local) throws Exception {
    String[] parts = coordinate.split(":", -1);
    if (parts.length != 2 || !coordinate.matches("[a-zA-Z0-9_.-]+:[a-zA-Z0-9_.-]+"))
      throw new IllegalArgumentException("Invalid Maven coordinate");
    byte[] payload =
        local == null
            ? Tools.get(
                CENTRAL + "/" + parts[0].replace('.', '/') + "/" + parts[1] + "/maven-metadata.xml",
                true)
            : (Files.exists(Path.of(local)) ? Files.readAllBytes(Path.of(local)) : null);
    if (payload == null) return List.of();
    var nodes = xml(payload).getElementsByTagName("version");
    if (nodes.getLength() == 0)
      throw new IllegalArgumentException("Maven metadata has no versions");
    var result = new ArrayList<String>();
    for (int i = 0; i < nodes.getLength(); i++) result.add(nodes.item(i).getTextContent());
    return result;
  }

  static boolean recoveryOnly(String repo, long run, int attempt, String event) throws Exception {
    if (run <= 0
        || attempt <= 0
        || !List.of("release", "push", "workflow_dispatch").contains(event))
      throw new IllegalArgumentException("Invalid recovery run");
    if (event.equals("workflow_dispatch")) return true;
    for (int prior = 1; prior < attempt; prior++) {
      JsonNode pages =
          Tools.gh(
              "repos/"
                  + repo
                  + "/actions/runs/"
                  + run
                  + "/attempts/"
                  + prior
                  + "/jobs?per_page=100",
              "--paginate",
              "--slurp");
      int jobs = 0;
      for (JsonNode page : pages)
        for (JsonNode job : page.path("jobs")) {
          jobs++;
          if (!job.path("name").asText().equals("Sign, privately stage and verify")) continue;
          if (job.path("status").asText().equals("completed")
              && job.path("conclusion").asText().equals("skipped")) continue;
          var stages = new ArrayList<JsonNode>();
          for (JsonNode step : job.path("steps"))
            if (step.path("name")
                .asText()
                .equals("Stage tested bytes, prove staged files and install a clean consumer"))
              stages.add(step);
          if (!job.path("status").asText().equals("completed")
              || stages.size() != 1
              || !stages.get(0).path("status").asText().equals("completed")
              || !stages.get(0).path("conclusion").asText().equals("skipped")) return true;
        }
      if (jobs == 0) throw new IllegalArgumentException("Cannot verify prior staging attempts");
    }
    return false;
  }

  static void run(Tools.Args args) throws Exception {
    String action = args.positional().get(0), repo = args.get("repo", Tools.repository());
    if (action.equals("parse-tag")) {
      var out = parseTag(args.positional().get(1));
      var meta = Tools.meta(out.get("plugin_dir"));
      if (!meta.name().equals(out.get("plugin"))
          || !meta.language().equals("java")
          || !meta.registry().equals("maven")
          || !meta.coordinate().equals("io.temporal:" + meta.name()))
        throw new IllegalArgumentException("Tag and plugin metadata differ");
      if (!FIRST.containsKey(meta.maturity()))
        throw new IllegalArgumentException("Unknown maturity");
      out.putAll(
          Map.of(
              "coordinate",
              meta.coordinate(),
              "root_api",
              meta.rootApi(),
              "smoke_imports",
              String.join(",", meta.imports()),
              "maturity",
              meta.maturity(),
              "github_prerelease",
              String.valueOf(
                  out.get("prerelease").equals("true") || meta.maturity().equals("pre-release"))));
      Tools.outputs(out, args.get("github-output"));
    } else if (action.equals("check-version-policy")) {
      var meta = Tools.meta(args.require("plugin-dir"));
      String raw = args.require("version");
      checkPolicy(raw, meta.maturity(), published(meta.coordinate(), args.get("registry-json")));
    } else if (action.equals("check-published-release")) {
      String tag = args.require("tag");
      var parsed = parseTag(tag);
      var meta = Tools.meta(parsed.get("plugin_dir"));
      if (!FIRST.containsKey(meta.maturity()))
        throw new IllegalArgumentException("Unknown maturity");
      JsonNode release =
          Tools.gh(
              "repos/" + repo + "/releases/tags/" + URLEncoder.encode(tag, StandardCharsets.UTF_8));
      boolean prerelease =
          parsed.get("prerelease").equals("true") || meta.maturity().equals("pre-release");
      if (!release.path("tag_name").asText().equals(tag)
          || !release.has("draft")
          || release.path("draft").asBoolean()
          || !release.has("prerelease")
          || release.path("prerelease").asBoolean() != prerelease
          || release.path("html_url").asText().isEmpty())
        throw new IllegalArgumentException("Published GitHub release differs");
      Tools.outputs(Map.of("url", release.path("html_url").asText()), args.get("github-output"));
    } else if (action.equals("check-java-stage-recovery")) {
      String event = args.require("event");
      boolean recover =
          recoveryOnly(
              repo,
              Long.parseLong(args.require("run-id")),
              Integer.parseInt(args.require("run-attempt")),
              event);
      Tools.outputs(
          Map.of(
              "recover_only",
              String.valueOf(recover),
              "restore_state",
              String.valueOf(recover && List.of("release", "push").contains(event))),
          args.get("github-output"));
    } else if (action.equals("release-notes")) {
      String tag = args.require("tag"), dir = args.require("plugin-dir");
      var parsed = parseTag(tag);
      var meta = Tools.meta(dir);
      Version current = Version.parse(parsed.get("version"));
      String previous = null;
      Version prior = null;
      for (String candidate :
          Tools.command("git", "tag", "--list", "java/" + meta.name() + "/v*").split("\n")) {
        if (candidate.isEmpty()) continue;
        Version v = Version.parse(candidate.substring(candidate.lastIndexOf("/v") + 2));
        if (v.compareTo(current) < 0 && (prior == null || v.compareTo(prior) > 0)) {
          previous = candidate;
          prior = v;
        }
      }
      String ref = args.get("ref", tag), range = previous == null ? ref : previous + ".." + ref;
      StringBuilder notes = new StringBuilder();
      if (previous == null)
        notes.append("First standalone release of `").append(meta.coordinate()).append("`.\n\n");
      notes.append("## What's changed\n\n");
      for (String line :
          Tools.command("git", "log", "--no-decorate", "--format=%h%x1f%s", range, "--", dir)
              .split("\n")) {
        if (line.isEmpty()) continue;
        String[] parts = line.split("\u001f", 2);
        notes.append("- ").append(parts[1]).append(" (").append(parts[0]).append(")\n");
      }
      if (previous != null)
        notes
            .append("\n**Full changelog**: https://github.com/")
            .append(repo)
            .append("/compare/")
            .append(previous)
            .append("...")
            .append(tag)
            .append('\n');
      if (current.rc() != null)
        notes.append(
            "\n"
                + "## Pre-release notes\n\n"
                + "This candidate is privately staged in Maven Central Portal.\n");
      Files.writeString(Path.of(args.require("output")), notes);
    } else if (List.of("bundle", "stage", "publish").contains(action)) Maven.run(args);
    else throw new IllegalArgumentException("Unknown Java release operation " + action);
  }

  public static void main(String[] args) {
    Tools.main(args, Release::run);
  }
}
