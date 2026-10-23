"""Where the integration keeps its files in a sandbox's home directory."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SandboxHome:
    """The integration's files in :attr:`Sandbox.home`, on the session's storage."""

    path: str

    @property
    def config(self) -> str:
        """The CLI's config dir, unless ``options.env`` sets ``CLAUDE_CONFIG_DIR``."""
        return f"{self.path}/claude-config"

    @property
    def plugins(self) -> str:
        """The plugins, each in a directory named by its index."""
        return f"{self.path}/plugins"

    @property
    def cli(self) -> CliFiles:
        """The files of the sandbox's CLI."""
        return CliFiles(f"{self.path}/cli")


@dataclass(frozen=True)
class CliFiles:
    """The files that carry the CLI's input, output, and state."""

    dir: str

    @property
    def stdin(self) -> str:
        """The directory of the CLI's input: ``1.jsonl``, ``2.jsonl``, ..., fed to it in order."""
        return self._file("stdin")

    @property
    def stdout(self) -> str:
        """The CLI's output, which the Activities Signal to the Workflow."""
        return self._file("stdout.jsonl")

    @property
    def log(self) -> str:
        """The CLI's stderr."""
        return self._file("cli.log")

    @property
    def pid(self) -> str:
        """The pid of the shell that runs the CLI, and the boot ID, once launched."""
        return self._file("pid")

    @property
    def new_pid(self) -> str:
        """Where :attr:`pid` is written before it is moved into place."""
        return self._file("pid.new")

    @property
    def exited(self) -> str:
        """The CLI's exit status, once it exited."""
        return self._file("exit")

    def _file(self, name: str) -> str:
        return f"{self.dir}/{name}"
