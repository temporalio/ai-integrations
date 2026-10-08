"""Standard-library-only renderer for a child shell call's recorded outcome."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    """Verify the receipt, then render only recorded bytes and status."""
    data = Path(sys.argv[1]).read_bytes()
    if hashlib.sha256(data).hexdigest() != sys.argv[2]:
        raise RuntimeError("Native result receipt changed")
    receipt = json.loads(data)
    output = receipt["output"]
    if receipt["is_error"]:
        sys.stdout.buffer.write(output.get("stdout", "").encode())
        sys.stderr.buffer.write(output.get("stderr", "").encode())
        code = output.get("exitCode", 1)
        sys.exit(code if isinstance(code, int) and 0 < code < 256 else 1)


if __name__ == "__main__":
    main()
