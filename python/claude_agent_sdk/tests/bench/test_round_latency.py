"""Opt-in: the hybrid benchmark's segment workload on this plugin (no timing asserts).

ROUND_BENCHMARK=1 runs it. ROUND_MODES (default "held,store"; add "-warm" to a mode
for warm_engines=4, as in "held-warm"), ROUND_TRIALS (3),
ROUND_COUNT (20), ROUND_OUT (a JSON file). Latency is the time between successive
model requests, as in the hybrid benchmark; wall time includes start and teardown.
ROUND_TOOL picks what Claude calls in each round: the durable echo tool (``echo``, the
default), or a Claude Code tool as a tool step: ``Bash``, ``Write``, or ``Read,Edit``
(a Read in the first round, in the segment, then an Edit in each round).
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import time
import uuid
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from temporalio.claude_agent_sdk import ClaudeAgentPlugin, ClaudeAgentSdkRunner
from temporalio.client import Client
from temporalio.worker import Worker
from tests.bench.store import TranscriptStore
from tests.bench.workflow import BuiltinBenchmarkWorkflow, SegmentBenchmarkWorkflow
from tests.bench.workload import segment_echo
from tests.helpers.fake_messages_api import FakeMessagesAPI, engine_env, history_of

pytestmark = [
    pytest.mark.timeout(1800),
    pytest.mark.skipif(
        os.environ.get("ROUND_BENCHMARK") != "1", reason="opt-in measurement"
    ),
]


def rounds(
    round_count: int, tool: str = "echo", work: Path = Path()
) -> FakeMessagesAPI:
    holder: list[FakeMessagesAPI] = []

    def call(n: int) -> dict[str, Any]:
        api = holder[0]
        if tool == "Bash":
            return api.call("Bash", {"command": f"echo {n}", "description": "echo"})
        if tool == "Write":
            target = str(work / f"f{n}.txt")
            return api.call("Write", {"file_path": target, "content": f"{n}\n"})
        if tool == "Read,Edit":
            notes = str(work / "notes.txt")
            if n == 0:
                return api.call("Read", {"file_path": notes})
            edit = {
                "file_path": notes,
                "old_string": f"v{n - 1}",
                "new_string": f"v{n}",
            }
            return api.call("Edit", edit)
        return api.tool_use("echo", {"n": n})

    def decide(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        if len(history) >= round_count:
            return [{"type": "text", "text": f"DONE {len(history)}"}]
        return [call(len(history))]

    api = FakeMessagesAPI(decide)
    holder.append(api)
    return api.start()


def model_requests(api: FakeMessagesAPI) -> list[dict[str, Any]]:
    return [
        r
        for r in api.requests
        if any(t["name"].startswith("mcp__durable__") for t in r.get("tools", []))
    ]


class CountingRunner(ClaudeAgentSdkRunner):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.starts = 0
        self.warm = 0

    async def _run_engine(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("warm") is None:
            self.starts += 1
        else:
            self.warm += 1
        return await super()._run_engine(*args, **kwargs)


async def test_rounds(client: Client, tmp_path: Path) -> None:
    modes = os.environ.get("ROUND_MODES", "held,store").split(",")
    tool = os.environ.get("ROUND_TOOL", "echo")
    trials = int(os.environ.get("ROUND_TRIALS", "3"))
    count = int(os.environ.get("ROUND_COUNT", "20"))
    measurements: list[dict[str, Any]] = []
    for trial in range(trials):
        for mode in modes:
            root = tmp_path / f"{mode}-{trial}"
            root.mkdir()
            (root / "notes.txt").write_text("v0\n", encoding="utf-8")
            api = rounds(count, tool, root)
            times: list[float] = []
            decide = api.decide

            def timed(body: dict[str, Any], decide: Any = decide) -> Any:
                times.append(time.monotonic())
                return decide(body)

            api.decide = timed
            queue = f"bench-{mode}-{uuid.uuid4().hex}"
            started = time.monotonic()
            try:
                warm = {"warm_engines": 4} if mode.endswith("-warm") else {}
                runner = CountingRunner(
                    session_store=(
                        TranscriptStore(root / "store.db")
                        if mode.startswith("store")
                        else None
                    ),
                    cwd=str(root),
                    env=engine_env(api, str(root / "cfg")),
                    **warm,
                )
                async with Worker(
                    client,
                    task_queue=queue,
                    workflows=[SegmentBenchmarkWorkflow, BuiltinBenchmarkWorkflow],
                    activities=[segment_echo],
                    plugins=[ClaudeAgentPlugin(runner)],
                ):
                    if tool == "echo":
                        answer = await client.execute_workflow(
                            SegmentBenchmarkWorkflow.run,
                            "work",
                            id=queue,
                            task_queue=queue,
                        )
                    else:
                        answer = await client.execute_workflow(
                            BuiltinBenchmarkWorkflow.run,
                            args=["work", tool],
                            id=queue,
                            task_queue=queue,
                        )
                elapsed = time.monotonic() - started
                assert answer == f"DONE {count}", answer
                assert len(times) == len(model_requests(api)) == count + 1
                assert api.errors == []
                latencies = [1000 * (b - a) for a, b in zip(times, times[1:])]
                measurements.append(
                    {
                        "mode": mode,
                        "tool": tool,
                        "trial": trial + 1,
                        "rounds": count,
                        "process_starts": runner.starts,
                        "warm_segments": runner.warm,
                        "model_requests": len(times),
                        "wall_seconds": elapsed,
                        "round_ms": latencies,
                        "median_ms": statistics.median(latencies),
                        "p95_ms": sorted(latencies)[
                            math.ceil(0.95 * len(latencies)) - 1
                        ],
                        "cli": await runner._engine_version(),
                    }
                )
            finally:
                api.stop()
    report = {
        "sdk": version("claude-agent-sdk"),
        "temporalio": version("temporalio"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "measurements": measurements,
    }
    target = Path(os.environ.get("ROUND_OUT", str(tmp_path / "rounds.json")))
    target.write_text(json.dumps(report, indent=2) + "\n")
    print("ROUND_BENCHMARK " + str(target), flush=True)
