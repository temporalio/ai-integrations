"""The shell scripts the Activities run in a sandbox. Each is a file in ``scripts/``."""

from __future__ import annotations

import functools
from pathlib import Path

from ._payloads import END_OF_INPUT

# Comes before every script.
_PRELUDE = f"""\
end_of_input={END_OF_INPUT}
# The sandbox's boot ID, or nothing if it has none. Stored with each pid, so a
# pid from before a restart is never taken for the CLI's.
boot_id() {{ cat /proc/sys/kernel/random/boot_id 2>/dev/null; }}
"""

_SCRIPTS = Path(__file__).parent / "scripts"


@functools.cache
def script(name: str) -> str:
    """The text of ``scripts/<name>.sh``, ready for ``sh -c``."""
    return _PRELUDE + (_SCRIPTS / f"{name}.sh").read_text()


def command(name: str, *args: str) -> list[str]:
    """The argv that runs script ``name`` with ``args`` as ``$1``, ``$2``, ..."""
    return ["sh", "-c", script(name), "sh", *args]
