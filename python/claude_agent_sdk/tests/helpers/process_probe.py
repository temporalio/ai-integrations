"""Disposable process tree for supervisor crash tests; no provider credentials."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    role, root, launcher, *modes = sys.argv[1:]
    mode = modes[0] if modes else "worker-loss"
    path = Path(root)
    if role == "worker":
        stderr = (path / "supervisor.stderr").open("w")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                launcher,
                str(os.getpid()),
                str(path / "workspace.lock"),
                sys.executable,
                __file__,
                "engine",
                root,
                launcher,
                mode,
            ],
            stderr=stderr,
        )
        with (path / "pids.jsonl").open("a") as log:
            log.write(json.dumps({"role": "supervisor", "pid": proc.pid}) + "\n")
    elif role == "engine":
        with (path / "pids.jsonl").open("a") as log:
            log.write(json.dumps({"role": role, "pid": os.getpid()}) + "\n")
        if mode == "double-fork":
            read_fd, write_fd = os.pipe()
            if os.fork():
                os.close(write_fd)
                assert os.read(read_fd, 1) == b"1"
                os.close(read_fd)
                return
            os.close(read_fd)
            os.setsid()
            if os.fork():
                os._exit(0)
            with (path / "pids.jsonl").open("a") as log:
                log.write(json.dumps({"role": "child", "pid": os.getpid()}) + "\n")
            with (path / "effects.log").open("a") as log:
                log.write("effect\n")
            os.write(write_fd, b"1")
            os.close(write_fd)
            role = "child"
        else:
            child = subprocess.Popen(
                [sys.executable, __file__, "child", root, launcher, mode],
                start_new_session=os.name != "nt",
            )
            with (path / "pids.jsonl").open("a") as log:
                log.write(json.dumps({"role": "child", "pid": child.pid}) + "\n")
            if mode == "exit-immediately":
                sys.exit(0)
    while True:
        if role == "child":
            with (path / "effects.log").open("a") as log:
                log.write("effect\n")
        if role == "engine" and (path / "exit-engine").exists():
            sys.exit(0)
        time.sleep(0.05)


if __name__ == "__main__":
    main()
