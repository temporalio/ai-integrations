"""Python distributions and installed-wheel checks, backed by each plugin's lock."""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@plugin_metadata//:plugins.bzl", "PLUGINS")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

def _dist_impl(ctx):
    version = ctx.attr._version[BuildSettingInfo].value
    stem = ctx.attr.coordinate.replace("-", "_") + "-" + version
    wheel = ctx.actions.declare_file("dist/" + stem + "-py3-none-any.whl")
    sdist = ctx.actions.declare_file("dist/" + stem + ".tar.gz")
    args = ctx.actions.args()
    args.add("--project", ctx.attr.project)
    args.add("--version", version)
    args.add("--wheel", wheel)
    args.add("--sdist", sdist)
    args.add_all(ctx.files.srcs, before_each = "--source")
    ctx.actions.run(
        executable = ctx.executable._builder,
        inputs = ctx.files.srcs,
        outputs = [wheel, sdist],
        arguments = [args],
        tools = [ctx.attr._builder[DefaultInfo].files_to_run],
        env = {"SOURCE_DATE_EPOCH": "315532800", "UV_NO_EDITABLE": "1"},
        mnemonic = "PythonDistribution",
    )
    return [DefaultInfo(files = depset([wheel, sdist])), OutputGroupInfo(wheel = depset([wheel]), sdist = depset([sdist]))]

python_dist = rule(
    implementation = _dist_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "coordinate": attr.string(mandatory = True),
        "project": attr.string(mandatory = True),
        "_version": attr.label(default = "//bazel:release_version"),
        "_builder": attr.label(default = "//python/_tools:build_dist", executable = True, cfg = "exec"),
    },
)

def _requirement_name(value):
    value = value.split(";")[0].split("[")[0]
    for separator in [">", "<", "=", "!", "~", " "]:
        value = value.split(separator)[0]
    return value.lower().replace("-", "_").replace(".", "_")

def python_integration(name = "check", test_deps = {}, test_data = {}):
    """Declare standard targets for a dependency-isolated Python plugin.

    Args:
        name: Name of the aggregate check suite.
        test_deps: Local Python imports needed by each test or conftest, relative to the plugin.
        test_data: Additional runfiles needed by an individual test.
    """
    package = native.package_name()
    meta = PLUGINS[package]
    project = meta["project"]
    hub = "python_" + meta["plugin"]["name"]
    roots = project["project"]["dependencies"] + project.get("dependency-groups", {}).get("dev", [])
    unconditional = []
    conditional = []
    for root in roots:
        if ";" not in root:
            unconditional.append(root)
        elif root.split(";", 1)[1].strip() == "python_version < '3.11'":
            if project["project"]["requires-python"].startswith(">=3.10"):
                conditional.append("@%s//%s" % (hub, _requirement_name(root)))
        else:
            fail("Declare the build selection for dependency marker: " + root)
    deps = ["@%s//%s" % (hub, dependency) for dependency in sorted({_requirement_name(root): True for root in unconditional})]
    lint_tools = ["ruff", "pyright", "basedpyright", "mypy", "pydocstyle"]
    runtime_deps = ["@%s//%s" % (hub, dependency) for dependency in sorted({_requirement_name(root): True for root in unconditional}) if dependency not in lint_tools]
    if conditional:
        deps = deps + select({"//bazel:python310": conditional, "//conditions:default": []})
        runtime_deps = runtime_deps + select({"//bazel:python310": conditional, "//conditions:default": []})
    namespace_wheels = ["@%s//%s:whl" % (hub, _requirement_name(root)) for root in roots if _requirement_name(root).startswith("temporalio_")]
    native.exports_files(["plugin.toml", "pyproject.toml", "uv.lock"])
    sources = native.glob(["src/**", "tests/**"], exclude = ["**/__pycache__/**", "**/*.pyc"])
    inputs = native.glob(["src/**", "README*.md", "LICENSE", "pyproject.toml"], exclude = ["**/__pycache__/**", "**/*.pyc"])
    py_library(
        name = "build",
        srcs = native.glob(["src/**/*.py"]),
        data = native.glob(["src/**/py.typed"]),
        imports = ["src"],
        deps = deps,
        visibility = ["//" + package + ":__subpackages__"],
    )
    python_dist(name = "dist", srcs = inputs, coordinate = meta["plugin"]["coordinate"], project = package)
    native.filegroup(name = "wheel", srcs = [":dist"], output_group = "wheel")
    test_files = native.glob(["tests/**/test_*.py"])
    checks = [(path[:-3], "test", path) for path in test_files]
    checks += [("lint", "lint", None), ("package_test", "package", None)]
    for test_name, mode, test_file in checks:
        if mode == "test":
            support = ["tests/__init__.py", "tests/conftest.py"]
            support += test_deps.get("tests/conftest.py", [])
            support += test_deps.get(test_file, [])
            check_inputs = [test_file] + support + test_data.get(test_file, [])
            if test_file == "tests/test_installed_matches_source.py":
                check_inputs += inputs
        elif mode == "package":
            check_inputs = ["tests/__init__.py", "tests/helpers/__init__.py", "tests/helpers/plugin_meta.py", "tests/helpers/provenance.py", "//:LICENSE", ":dist"]
            check_inputs += test_deps.get("tests/helpers/__init__.py", [])
        else:
            check_inputs = sources + inputs
        py_test(
            name = test_name,
            srcs = ["//python/_tools:installed_runner.py"],
            main = "//python/_tools:installed_runner.py",
            args = ["--project", package, "--mode", mode, "--wheel", "$(rlocationpath :wheel)", "--sdk-wheel", "$(rlocationpath @%s//temporalio:whl)" % hub] + [value for wheel in namespace_wheels for value in ["--extra-wheel", "$(rlocationpath %s)" % wheel]] + (["--test", test_file] if test_file else []),
            data = sorted({path: True for path in check_inputs + ["pyproject.toml", "plugin.toml", ":wheel", "@%s//temporalio:whl" % hub, "//python/_shared:config", "//bazel/temporal:cli_python", "//bazel:node"] + namespace_wheels}),
            deps = (deps if mode == "lint" else runtime_deps) + ["@rules_python//python/runfiles"],
            env = {"TEMPORAL_CLI_PATH": "$(rlocationpath //bazel/temporal:cli_python)", "UV_NO_EDITABLE": "1", "NODE_BINARY": "$(rlocationpath //bazel:node)"},
            legacy_create_init = False,
            size = "large",
            timeout = "long",
            tags = ["no-remote-exec"] if mode == "test" else [],
        )
    native.test_suite(name = "tests", tests = [":" + path[:-3] for path in test_files])
    native.test_suite(name = name, tests = [":lint", ":tests", ":package_test"])
