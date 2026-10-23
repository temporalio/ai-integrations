"""Names and payloads shared by the Workflow and the sandbox Activities."""

from __future__ import annotations

from dataclasses import dataclass, field

CONNECT = "claude-sandbox-connect"
LAUNCH = "claude-sandbox-launch"
WRITE_STDIN = "claude-sandbox-write-stdin"
WATCH_OUTPUT = "claude-sandbox-watch-output"
CLOSE = "claude-sandbox-close"

# The line that ends the CLI's stdin, as the Claude Agent SDK's end_input closes it.
END_OF_INPUT = "temporal-end-of-input"


# The kinds of line in the CLI's stdout. Each Signal of stdout is named for the
# kind of line it carries, so the Temporal UI shows what it is.
OUTPUT_KINDS = (
    "result",
    "control_request",
    "control_response",
    "tool_use",
    "tool_result",
    "assistant",
    "other",
)


def transcripts(config_dir: str) -> str:
    """The glob of the transcripts a CLI resumes from: the session's and its subagents'."""
    return f"{config_dir}/projects/**/*.jsonl"


@dataclass
class SandboxSession:
    """A started sandbox, and the CLI in it once launched."""

    sandbox: str
    config_dir: str
    """The CLI's config dir in the sandbox. It holds the session's files."""
    plugin_dirs: list[str] = field(default_factory=list[str])
    cli_pid: int = 0
    """The pid of the shell that runs the CLI and waits for it."""
    cli_boot: str = ""
    """The sandbox's boot ID when the CLI started, or empty if it has none."""
    first_execution_run_id: str = ""


@dataclass
class ConnectRequest:
    """Activity input to start a sandbox for a session and write the plugins into it."""

    sandbox: str
    first_execution_run_id: str
    """Every sandbox started for the same session shares its storage."""
    env: dict[str, str] = field(default_factory=dict[str, str])
    """The CLI's environment. The Activity adds the config dir."""
    plugins: list[str] = field(default_factory=list[str])


@dataclass
class CloseRequest:
    """Activity input to end the session's sandbox."""

    sandbox: str
    first_execution_run_id: str


@dataclass
class LaunchRequest:
    """Activity input to start the CLI in a connected sandbox."""

    session: SandboxSession
    cli_args: list[str]


@dataclass
class StdinWrite:
    """Activity input to write ``text`` as the CLI's stdin file ``number``."""

    session: SandboxSession
    number: int
    text: str


@dataclass
class WatchRequest:
    """Activity input to Signal the CLI's stdout to the Workflow, from byte ``offset``."""

    session: SandboxSession
    offset: int
    stream_topic: str | None = None
    """The Workflow Stream topic stream_event lines are published to, if any."""


@dataclass
class CliOutput:
    """A line of the CLI's stdout, which ends at byte ``end``.

    ``start`` is where the output the Workflow doesn't have yet begins: any
    lines before this one went to a Workflow Stream.
    """

    # First, so the Temporal UI shows it at the top of the Signal's input.
    summary: str = field(default="", kw_only=True)
    """What the line is, such as ``tool_use: Bash``."""
    start: int
    end: int
    line: str
