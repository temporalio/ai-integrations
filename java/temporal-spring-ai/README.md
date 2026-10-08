# Temporal Spring AI

Public Preview integration for Spring AI 2.0.x and Spring Boot 4. Model calls
execute as Temporal Activities; Activity, Nexus, side-effect, and deterministic
tools execute through the ChatClient tool-calling advisor on the workflow thread.

The coordinate remains `io.temporal:temporal-spring-ai` on
[Maven Central](https://central.sonatype.com/artifact/io.temporal/temporal-spring-ai).
The candidate planned here is **1.41.0-RC1**, staged privately in Central Portal.
CI tests Java 17 and 25; published bytecode targets Java 17.

## Requirements and installation

- Java 17 or later.
- Spring AI 2.0.1 and Spring Boot 4.0.8; Boot 4.1.1 is also tested.
- Temporal Java SDK and `temporal-spring-boot-starter` 1.40.0 or later.
- A Spring AI provider starter for your chosen model. This integration supplies no provider.

After the standalone final release is available:

```groovy
implementation 'io.temporal:temporal-spring-ai:1.41.0'
implementation 'io.temporal:temporal-spring-boot-starter:1.40.0'
implementation 'io.temporal:temporal-sdk:1.40.0'
```

Inside a workflow, construct the client with `TemporalChatClient.builder(ActivityChatModel.forDefault())`
and register tools with `defaultTools(...)`. The entry points `TemporalChatClient`,
`ActivityChatModel`, and Temporal tool annotations retain their names and packages.
Model calls run as Activities; tool calls remain durable Temporal operations.

## Moving from Spring AI 1

Version 1.41 targets Spring AI 2 / Boot 4 exclusively. Spring AI 1 / Boot 3 applications
remain on their existing 1.40.x artifacts. Align your application with the Spring AI 2
and Boot 4 BOMs; Spring AI 2 uses immutable options and Jackson 3. Pass option builders
to ChatClient, for example `.defaultOptions(OpenAiChatOptions.builder().reasoningEffort("high"))`.
`ActivityChatModel.call(...)` returns a raw model response; use ChatClient's advisor
loop to execute tools. The removed `internalToolExecutionEnabled` flag is unnecessary.

Do not move in-flight Spring AI 1 workflows onto these workers: the upgrade does not
provide replay compatibility with Spring AI 1 histories. Use separate worker deployments
or complete the existing executions before switching versions.

Streaming and tool context remain unsupported. Provider-specific options survive
Activity serialization through their public builders; callbacks are transported as
definitions and execute only in the workflow. Retry, per-model Activity options,
response metadata, and media-size guards retain their behavior. Optional vector-store,
embedding, and MCP modules are detected through Spring auto-configuration.

The [archived Spring AI 1 usage guide](https://github.com/temporalio/ai-integrations/blob/main/java/temporal-spring-ai/_upstream/README.md)
records the imported implementation. Maintenance is now owned by ai-integrations;
there is no active upstream sync relationship.

## Development

```bash
./gradlew spotlessCheck test stageDist
```

The full Java/OS matrix runs for each supported Boot version, using committed locks.
Deliberately update locks with `./gradlew resolveAndLockAll --write-locks`
(and `-PspringBootVersion=4.1.1` for that compatibility lane).

Nightly CI and manual runs with `latest-deps=true` test stable releases in Temporal
1.x, Spring AI 2.x, and Spring Boot 4.x.
To run that check locally:

```bash
./gradlew -PdependencyMode=latest resolveAndLockAll --write-locks --refresh-dependencies
./gradlew -PdependencyMode=latest spotlessCheck test stageDist
```

Latest mode writes a separate ignored lockfile and reuses its selected versions
through testing and building. Ordinary CI and releases use the committed locks.

Committed development builds use `0.0.0`. CI supplies `-PreleaseVersion=<version>`
from an immutable release tag before testing and building. No placeholder versions
or snapshots are published.

TRANSITION(sdk-cutover): final standalone publication remains disabled until
sdk-java stops publishing this coordinate. The imported implementation and tests,
as well as the migration adaptations, are relicensed under the repository's MIT
license. Historical upstream commits retain their original licensing records.
