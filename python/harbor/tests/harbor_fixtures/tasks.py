"""Tiny harbor tasks, written to a temporary directory per test.

Each task's solution and verifier are bash scripts that find the trial's
``/logs`` through ``$HARBOR_LOGS`` (see ``local_env``), so they would also run
unchanged in a real container.
"""

from __future__ import annotations

from pathlib import Path

LOCAL_ENV = "tests.harbor_fixtures.local_env:LocalEnvironment"

_TASK_TOML = """\
version = "1.0"

[verifier]
timeout_sec = 60.0

[agent]
timeout_sec = 60.0
"""

_PRELUDE = 'LOGS="${HARBOR_LOGS:-/logs}"\n'

_WRITE_ANSWER = 'mkdir -p "$LOGS/artifacts" && echo hello > "$LOGS/artifacts/out.txt"\n'

_GRADE = """\
mkdir -p "$LOGS/verifier"
if grep -q hello "$LOGS/artifacts/out.txt" 2>/dev/null; then
  echo 1 > "$LOGS/verifier/reward.txt"
else
  echo 0 > "$LOGS/verifier/reward.txt"
fi
"""


def _task(root: Path, name: str, *, solve: str, test: str) -> Path:
    task = root / name
    for sub in ("solution", "tests", "environment"):
        (task / sub).mkdir(parents=True, exist_ok=True)
    (task / "task.toml").write_text(_TASK_TOML)
    (task / "instruction.md").write_text(
        "Write the word hello to the artifacts file.\n"
    )
    for path, body in (
        (task / "solution" / "solve.sh", solve),
        (task / "tests" / "test.sh", test),
    ):
        path.write_text("#!/bin/bash\n" + _PRELUDE + body)
        path.chmod(0o755)
    return task


def passing(root: Path, name: str = "answers") -> Path:
    """The agent's solution is correct: reward 1."""
    return _task(root, name, solve=_WRITE_ANSWER, test=_GRADE)


def failing(root: Path, name: str = "stays-silent") -> Path:
    """The agent writes nothing: reward 0."""
    return _task(root, name, solve="true\n", test=_GRADE)


def no_reward(root: Path, name: str = "no-reward") -> Path:
    """The verifier writes no reward, which harbor records as an error."""
    return _task(root, name, solve=_WRITE_ANSWER, test="true\n")


def flaky(root: Path, failures: int, name: str = "flaky") -> Path:
    """The verifier writes no reward on its first ``failures`` runs, then grades.

    The run count lives beside the task, outside any trial directory, so it
    survives a retry that sets the previous attempt's directory aside.
    """
    counter = root / f"{name}.runs"
    test = (
        f'n=$(( $(cat "{counter}" 2>/dev/null || echo 0) + 1 )); echo $n > "{counter}"\n'
        f"if [ $n -le {failures} ]; then exit 0; fi\n" + _GRADE
    )
    return _task(root, name, solve=_WRITE_ANSWER, test=test)


def slow(root: Path, seconds: float, name: str = "slow") -> Path:
    """The agent's solution is correct but takes ``seconds`` to write."""
    return _task(root, name, solve=f"sleep {seconds}\n" + _WRITE_ANSWER, test=_GRADE)
