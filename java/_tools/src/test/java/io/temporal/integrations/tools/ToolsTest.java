package io.temporal.integrations.tools;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

public class ToolsTest {
  @Test
  void canonicalVersions() {
    assertTrue(Release.Version.parse("1.0.0-RC2").compareTo(Release.Version.parse("1.0.0")) < 0);
    for (String value : new String[] {"01.0.0", "1.0", "1.0.0-rc.1", "1.0.0+build"})
      assertThrows(IllegalArgumentException.class, () -> Release.Version.parse(value));
  }

  @Test
  void forwardPolicy() {
    Release.checkPolicy("0.1.0-RC1", "public-preview", java.util.List.of());
    assertThrows(
        IllegalArgumentException.class,
        () -> Release.checkPolicy("1.0.0", "public-preview", java.util.List.of()));
    assertThrows(
        IllegalArgumentException.class,
        () -> Release.checkPolicy("0.1.0-RC1", "public-preview", java.util.List.of("0.1.0")));
  }

  @Test
  void xmlRejectsExternalEntities() {
    assertThrows(
        Exception.class,
        () ->
            Release.xml(
                "<!DOCTYPE x [<!ENTITY e SYSTEM 'file:///etc/passwd'>]><x>&e;</x>".getBytes()));
  }

  @Test
  void signedBundleRejectsDifferentOrMissingBytes() throws Exception {
    var raw = java.util.Map.of("io/temporal/example/1.0.0/example-1.0.0.jar", "tested".getBytes());
    var contents = new java.util.TreeMap<String, byte[]>();
    String name = raw.keySet().iterator().next();
    contents.put(name, raw.get(name));
    contents.put(name + ".asc", "-----BEGIN PGP SIGNATURE-----\ntest".getBytes());
    for (String file : java.util.List.of(name, name + ".asc"))
      for (String hash : Maven.HASHES)
        contents.put(
            file + "." + Maven.checksumSuffix(hash),
            Maven.digest(contents.get(file), hash).getBytes());
    var directory = java.nio.file.Files.createTempDirectory("bundle-test-");
    try {
      var bundle = directory.resolve("bundle.zip");
      Maven.writeBundle(bundle, contents);
      assertEquals(Maven.names(raw), Maven.validateBundle(bundle, raw).keySet());
      for (String file : new java.util.ArrayList<>(contents.keySet())) {
        byte[] original = contents.get(file);
        contents.put(file, "corrupt".getBytes());
        Maven.writeBundle(bundle, contents);
        assertThrows(IllegalArgumentException.class, () -> Maven.validateBundle(bundle, raw));
        contents.put(file, original);
      }
      contents.put("extra", "unexpected".getBytes());
      Maven.writeBundle(bundle, contents);
      assertThrows(IllegalArgumentException.class, () -> Maven.validateBundle(bundle, raw));
    } finally {
      PackageTool.delete(directory);
    }
  }

  @Test
  void portalDeploymentIdsAreCanonical() {
    String id = "e81da5dc-2a9a-4fd0-b668-09721605de52";
    assertEquals(id, Maven.deploymentId(id));
    for (String invalid : new String[] {id.toUpperCase(), "1-1-1-1-1", "not-a-deployment"})
      assertThrows(IllegalArgumentException.class, () -> Maven.deploymentId(invalid));
  }

  @Test
  void keyPayloadPreservesArmorAndAcceptsBase64() {
    String armor = "-----BEGIN PGP PRIVATE KEY BLOCK-----\nsecret\n";
    assertArrayEquals(armor.getBytes(), Maven.keyPayload(armor));
    assertArrayEquals(
        armor.getBytes(),
        Maven.keyPayload(java.util.Base64.getEncoder().encodeToString(armor.getBytes())));
    assertThrows(IllegalArgumentException.class, () -> Maven.keyPayload("%%%"));
  }

  @Test
  void registryBytesAreComparedWithoutMutation() throws Exception {
    var contents = java.util.Map.of("artifact", "tested".getBytes());
    Maven.verifyRemote(contents, name -> "tested".getBytes(), 1);
    assertThrows(
        java.io.IOException.class,
        () -> Maven.verifyRemote(contents, name -> "different".getBytes(), 1));
    assertThrows(java.io.IOException.class, () -> Maven.verifyRemote(contents, name -> null, 1));
  }

  @Test
  void savedDeploymentAllowsAnEmptyDispatchField() {
    String id = "e81da5dc-2a9a-4fd0-b668-09721605de52";
    Maven.requestedDeployment("", id);
    Maven.requestedDeployment(null, id);
    Maven.requestedDeployment(id, id);
    assertThrows(IllegalArgumentException.class, () -> Maven.requestedDeployment("another", id));
  }

  @Test
  void workflowRunIdsUse64Bits() throws Exception {
    assertTrue(Release.recoveryOnly("example/repository", 30000000000L, 1, "workflow_dispatch"));
  }
}
