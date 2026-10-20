"""Native TypeScript compilation, AVA tests, ESLint and npm distribution actions."""

load("@aspect_rules_js//js:defs.bzl", "js_library", "js_test")
load("@aspect_rules_js//js:providers.bzl", "JsInfo")
load("@aspect_rules_ts//ts:defs.bzl", "ts_config", "ts_project")
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@plugin_metadata//:plugins.bzl", "PLUGINS")

def _dist_impl(ctx):
    version = ctx.attr._version[BuildSettingInfo].value
    output = ctx.actions.declare_file("dist/" + ctx.attr.coordinate.replace("@", "").replace("/", "-") + "-" + version + ".tgz")
    args = ctx.actions.args()
    args.add_all(["pack", "--project", ctx.label.package, "--compiled", ctx.bin_dir.path + "/" + ctx.label.package + "/lib", "--declarations", ctx.bin_dir.path + "/" + ctx.label.package + "/public-types", "--version", version, "--output", output.path])
    compiled = [file for target in ctx.attr.compiled for file in depset(transitive = [target[JsInfo].sources, target[JsInfo].types]).to_list() if file.short_path.startswith(ctx.label.package + "/lib/")]
    inputs = depset(ctx.files.srcs + compiled + ctx.files.declarations)
    ctx.actions.run(executable = ctx.executable._packer, arguments = [args], inputs = inputs, tools = [ctx.attr._packer[DefaultInfo].files_to_run], outputs = [output], env = {"BAZEL_BINDIR": ctx.bin_dir.path, "JS_BINARY__NO_CD_BINDIR": "1"}, mnemonic = "NpmDistribution")
    return [DefaultInfo(files = depset([output]))]

npm_dist = rule(implementation = _dist_impl, attrs = {
    "srcs": attr.label_list(allow_files = True),
    "compiled": attr.label_list(),
    "declarations": attr.label(mandatory = True),
    "coordinate": attr.string(),
    "_packer": attr.label(default = "//typescript/_tools:package", cfg = "exec", executable = True),
    "_version": attr.label(default = "//bazel:release_version"),
})

def _windows_test_paths_impl(ctx):
    failures = []

    # Bazel's native Windows wrapper invokes the executable inside its runfiles
    # tree. Check that complete path against MAX_PATH on every host platform.
    prefix = "D:/_bazel/execroot/_main/bazel-out/x64_windows-fastbuild-node24/bin/"
    for target in ctx.attr.tests:
        executable = target[DefaultInfo].files_to_run.executable.short_path
        if not executable.endswith(".bat"):
            executable += ".bat"
        path = prefix + executable + ".runfiles/_main/" + executable
        if len(path) >= 260:
            failures.append("%s: Windows test launcher path is %d characters: %s" % (target.label, len(path), path))
    return [AnalysisTestResultInfo(success = not failures, message = "\n".join(failures))]

_windows_test_paths_test = rule(
    implementation = _windows_test_paths_impl,
    attrs = {"tests": attr.label_list(mandatory = True)},
    test = True,
    analysis_test = True,
)

def typescript_module(name, srcs, deps = [], data = []):
    """Compile one module (or a group of mutually importing modules).

    Args:
        name: Module target name.
        srcs: TypeScript source files.
        deps: Direct module and npm dependencies, including modules loaded by path.
        data: Runtime fixtures used by the module.
    """
    compilation = name.replace("/", "__") + "_compile"
    module_compilations = [dep[1:].replace("/", "__") + "_compile" for dep in deps if dep.startswith(":") and not dep.startswith(":node_modules")]
    ts_project(
        name = compilation,
        srcs = srcs,
        # Internal declarations connect compilation units. Public declarations
        # apply the plugin's stripInternal setting separately at packaging time.
        # Override the package-wide include inherited from tsconfig.json. On
        # Windows, all sources are visible in the shared execution directory;
        # otherwise each compiler emits files owned by other module actions.
        tsconfig = {
            "compilerOptions": {"stripInternal": False},
            "include": [],
            # Resolve local imports to the referenced module's declarations,
            # even when its .ts sources are visible outside a sandbox.
            "references": [{"path": "./tsconfig_" + dep + ".json"} for dep in module_compilations],
        },
        extends = ":tsconfig",
        out_dir = "lib",
        root_dir = "src",
        composite = True,
        incremental = True,
        declaration = True,
        source_map = True,
        deps = [":node_modules/@types/node"] + deps + [":" + dep for dep in module_compilations],
        data = srcs + data,
    )

    # Relative imports in another project's src/ must resolve declarations at
    # the same paths. Keep the published declarations in lib/ as well.
    declarations = []
    for index, source in enumerate(srcs):
        declaration = compilation + "_declaration_" + str(index)
        copy_file(
            name = declaration,
            src = "lib/" + source[len("src/"):-3] + ".d.ts",
            out = source[:-3] + ".d.ts",
            allow_symlink = True,
        )
        declarations.append(":" + declaration)
    js_library(name = name, srcs = declarations + [":" + compilation])

def typescript_integration(modules, test_modules, name = "check", test_data = {}):
    """Declare native TypeScript integration targets.

    Args:
        name: Name of the aggregate check suite.
        modules: Production module targets.
        test_modules: Test and test support module targets.
        test_data: Additional dependencies for tests that load files dynamically.
    """
    package = native.package_name()
    manifest = PLUGINS[package]["project"]
    js_library(name = "package_metadata", srcs = ["package.json"])

    # Declaration emit uses the package boundary and dependency names to produce
    # portable type references, including types inferred from SDK return values.
    ts_config(name = "tsconfig", src = "tsconfig.json", deps = [":package_metadata", ":node_modules/@tsconfig/node20", "//typescript/_shared:tsconfig"])
    js_library(name = "build", srcs = modules)
    js_library(name = "test_build", srcs = test_modules)
    ts_project(
        name = "public_types",
        srcs = native.glob(["src/**/*.ts"], exclude = ["src/__tests__/**"]),
        tsconfig = {"compilerOptions": {"outDir": "public-types"}, "include": []},
        extends = ":tsconfig",
        out_dir = "public-types",
        root_dir = "src",
        composite = True,
        incremental = True,
        declaration = True,
        emit_declaration_only = True,
        source_map = False,
        deps = modules,
        tags = ["manual"],
    )
    patterns = [pattern.replace("./lib/", "src/").replace(".js", ".ts") for pattern in manifest["ava"]["files"]]
    tests = native.glob(patterns)
    launchers = []
    for source in tests:
        test_name = "tests/" + source[len("src/__tests__/"):-3]

        # rules_js repeats the target name in the launcher and runfiles paths.
        # A stable short name avoids Windows' limit without changing per-module
        # public test labels or invalidating other tests when a file is added.
        launcher = "_t_" + str(hash(test_name))
        js_test(name = launcher, entry_point = "//typescript/_tools:lib/test-runner.js", args = [source.replace("src/", "lib/", 1)[:-3] + ".js"], data = [":" + source[len("src/"):-3], ":package_metadata", ":node_modules/ava", "//bazel/temporal:cli", "//typescript/_tools:test_runtime", ":tsconfig"] + test_data.get(source, []), no_copy_to_bin = ["//bazel/temporal:cli"], chdir = package, env = {"TEMPORAL_CLI_RUNFILE": "$(rlocationpath //bazel/temporal:cli)"}, tags = ["no-remote-exec"], size = "large", timeout = "long")
        native.test_suite(name = test_name, tests = [":" + launcher])
        launchers.append(":" + launcher)
    _windows_test_paths_test(name = "test_paths", tests = launchers)
    native.test_suite(name = "tests", tests = [":tests/" + source[len("src/__tests__/"):-3] for source in tests] + [":test_paths"])
    js_test(name = "lint", entry_point = "//typescript/_tools:lib/lint-runner.js", args = native.glob(["src/**/*.ts"]), data = [":build", ":test_build", ":node_modules", ":tsconfig", ":package_metadata", "//typescript/_tools:lint_runtime", "//typescript/_shared:config"], chdir = package)
    npm_dist(name = "dist", coordinate = manifest["name"], compiled = [":build"], declarations = ":public_types", srcs = native.glob(["src/**", "README.md", "LICENSE", "package.json"], exclude = ["src/__tests__/**"]))
    native.test_suite(name = name, tests = [":lint", ":tests"])
