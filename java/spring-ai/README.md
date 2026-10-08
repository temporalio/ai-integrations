# Temporal Spring AI

Public Preview integration that executes model calls as Temporal Activities and
dispatches Activity, Nexus, side-effect, and deterministic tools from Workflows.

Coordinate: `io.temporal:spring-ai`. The first Public Preview release is planned
as `0.1.0`, with `0.1.0-RC1` staged privately in Central Portal first. Existing
SDK releases use `io.temporal:temporal-spring-ai` on
[Maven Central](https://central.sonatype.com/artifact/io.temporal/temporal-spring-ai).
After `io.temporal:spring-ai:0.1.0` is published, sdk-java publishes a relocation
POM under the next old-coordinate version pointing to it.
This initial import retains Spring AI 1.1.0, Spring Boot 3.5.12, and Java 17+.
Consumers supply the Temporal SDK and `temporal-spring-boot-starter` separately.
CI tests Java 17 and 25; published bytecode targets Java 17.

See the [imported usage guide](https://github.com/temporalio/ai-integrations/blob/main/java/spring-ai/_upstream/README.md).
Imported source and tests remain upstream-owned until the ownership handoff. The
archived installation instructions use the old Maven coordinate; standalone
releases use `io.temporal:spring-ai`.

## Development

```bash
./gradlew spotlessCheck test stageDist
```

Dependencies are locked per supported Spring Boot version. To deliberately update
a lock, run `./gradlew resolveAndLockAll --write-locks` (with
`-PspringBootVersion=<version>` for a compatibility lane).

Nightly CI and manual runs with `latest-deps=true` test stable releases in Temporal
1.x, Spring AI 1.x, and Spring Boot 3.x.
To run that check locally:

```bash
./gradlew -PdependencyMode=latest resolveAndLockAll --write-locks --refresh-dependencies
./gradlew -PdependencyMode=latest spotlessCheck test stageDist
```

Latest mode writes a separate ignored lockfile and reuses its selected versions
through testing and building. Ordinary CI and releases use the committed locks.

Committed development builds use `0.0.0`. CI supplies `-PreleaseVersion=<version>`
from an immutable release tag before testing and building; do not commit release
version bumps. No `0.0.0` artifacts are published.

TRANSITION(sdk-cutover): final standalone publication awaits the ownership
handoff and an agreed SDK cutover plan. Publish the new package before sdk-java
publishes the old coordinate's relocation POM. The imported implementation and tests,
as well as the migration adaptations, are relicensed under the repository's MIT
license. Historical upstream commits retain their original licensing records.
