"""Expose the selected host Go SDK for explicit developer operations."""

def _sdk_impl(ctx):
    sdk = ctx.toolchains["@rules_go//go:toolchain"].sdk
    return [DefaultInfo(files = depset([sdk.go]), runfiles = ctx.runfiles([sdk.go], transitive_files = depset(transitive = [sdk.headers, sdk.srcs, sdk.libs, sdk.tools])))]

go_command = rule(implementation = _sdk_impl, toolchains = ["@rules_go//go:toolchain"])
