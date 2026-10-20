"""Bazel Java publications with separately locked compilation and runtime graphs."""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@contrib_rules_jvm//java:defs.bzl", "create_jvm_test_suite", "java_junit5_test", "java_library")
load("@rules_java//java:defs.bzl", "JavaInfo")

def _short_junit5_test(name, **kwargs):
    # The JVM's Windows agent loader cannot open paths beyond MAX_PATH. Using
    # the class basename keeps the runfiles directory and agent path short.
    name = name.rsplit("/", 1)[-1]
    java_junit5_test(name = name, **kwargs)
    return name

def java_junit5_test_suite(name, srcs, package, **kwargs):
    """Create JUnit 5 tests with short executable and runfiles paths.

    Args:
        name: Name of the aggregate test suite.
        srcs: Java test and helper sources.
        package: Root Java package for the test classes.
        **kwargs: Arguments forwarded to the test suite generator.
    """
    create_jvm_test_suite(
        name = name,
        srcs = srcs,
        package = package,
        test_suffixes = ["Test.java"],
        define_library = java_library,
        define_test = _short_junit5_test,
        **kwargs
    )

def _dist_impl(ctx):
    out = ctx.actions.declare_directory("dist")
    jars = ctx.attr.library[JavaInfo].transitive_compile_time_jars
    args = ctx.actions.args()
    args.add_all(["build", "--project", ctx.label.package, "--version", ctx.attr._version[BuildSettingInfo].value, "--spring-boot", ctx.attr._boot[BuildSettingInfo].value, "--jar", ctx.attr.library[JavaInfo].outputs.jars[0].class_jar.path, "--output", out.path])
    args.add_joined("--classpath", jars, join_with = ctx.configuration.host_path_separator)
    ctx.actions.run(executable = ctx.executable._tool, inputs = depset(ctx.files.srcs, transitive = [ctx.attr.library[DefaultInfo].files, jars]), tools = [ctx.attr._tool[DefaultInfo].files_to_run], outputs = [out], arguments = [args], mnemonic = "MavenDistribution")
    return [DefaultInfo(files = depset([out]))]

java_dist = rule(implementation = _dist_impl, attrs = {
    "library": attr.label(providers = [JavaInfo]),
    "srcs": attr.label_list(allow_files = True),
    "_tool": attr.label(default = "//java/_tools:package", cfg = "exec", executable = True),
    "_version": attr.label(default = "//bazel:release_version"),
    "_boot": attr.label(default = "//bazel:spring_boot"),
})
