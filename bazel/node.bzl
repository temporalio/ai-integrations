"""Expose only the selected Node executable to Python static-analysis tools."""

def _node_impl(ctx):
    node = ctx.toolchains["@rules_nodejs//nodejs:runtime_toolchain_type"].nodeinfo.node
    return [DefaultInfo(files = depset([node]), runfiles = ctx.runfiles([node]))]

node_command = rule(implementation = _node_impl, toolchains = ["@rules_nodejs//nodejs:runtime_toolchain_type"])
