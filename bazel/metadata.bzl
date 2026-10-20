"""Read plugin metadata once for build rules and native CI executables."""

load("@toml.bzl", "toml")

def _matrix(meta):
    versions = meta["ci"]["runtime-versions"]
    cells = []
    if versions[0] != versions[-1]:
        cells.append({"os": "ubuntu-latest", "runtime": versions[0], "dist": False})
    cells.append({"os": "ubuntu-latest", "runtime": versions[-1], "dist": True})
    cells.extend([{"os": os, "runtime": versions[-1], "dist": False} for os in ["macos-latest", "windows-latest"]])
    boots = meta["ci"].get("spring-boot-versions", [])
    if boots:
        cells = [dict(cell, spring_boot = boot, dist = cell["dist"] and i == 0) for i, boot in enumerate(boots) for cell in cells]
    return {"include": cells}

def _metadata_impl(ctx):
    root = ctx.path(ctx.attr.root).dirname
    plugins = {}
    for language in ["python", "typescript", "java", "go"]:
        directory = root.get_child(language)
        ctx.watch(directory)
        for child in directory.readdir():
            if child.basename.startswith(("_", ".")) or not child.is_dir:
                continue
            manifest = child.get_child("plugin.toml")
            if not manifest.exists:
                continue
            meta = toml.decode(ctx.read(manifest))
            key = language + "/" + child.basename
            meta["path"] = key
            meta["matrix"] = _matrix(meta)
            project = child.get_child("pyproject.toml")
            if language == "python":
                meta["project"] = toml.decode(ctx.read(project))
            elif language == "typescript":
                meta["project"] = json.decode(ctx.read(child.get_child("package.json")))
            plugins[key] = meta
            ctx.file(language + "_" + child.basename + ".json", json.encode_indent(meta))
    ctx.file("plugins.json", json.encode_indent(plugins))
    ctx.file("plugins.bzl", "PLUGINS = " + repr(plugins) + "\n")
    ctx.file("BUILD.bazel", 'exports_files(glob(["*.json", "*.bzl"]), visibility = ["//visibility:public"])\n')

plugin_metadata = repository_rule(
    implementation = _metadata_impl,
    attrs = {"root": attr.label(mandatory = True, allow_single_file = True)},
    configure = True,
)
