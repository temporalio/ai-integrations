"""Stand-in Claude Code CLI for ``ClaudeSDKClient``, modeled on CLI 2.1.294.

Speaks stream-json, control requests, SDK MCP and EOF/SIGTERM teardown, with no network and
no credentials. The JSON scenario file is the first argument. Its main keys are ``turn`` /
``turns`` (stdout lines, optionally per prompt prefix), ``plans`` (the measured crash/resume
model, see ``Transcript`` and ``plan_steps``), ``crash_at`` / ``crash_mode`` (see
``trigger``), ``session_mode``, and process controls such as ``exit_code``, ``hang_*``,
``child`` and ``record``. ``listen_while_streaming`` answers an interrupt after each ``sleep``
action, as a real CLI does mid-stream. Lines ``{"fake": "mcp_calls" | "native_call" | "sleep", ...}`` are
actions, not output; ``{{P}}``, ``{{RESULT:<tool_use_id>}}`` and ``{{RUN}}`` are substituted.
"""

from __future__ import annotations

import json
import os
import select
import shlex
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from claude_agent_sdk import project_key_for_directory
from temporalio.claude_agent_sdk._native import content_blocks

SERVER = "temporal"
# What 2.1.294's default permission mode runs without asking when no hook decided.
_AUTO_READS = frozenset({"Read", "Glob", "Grep"})


def _flag(argv: list[str], name: str) -> str | None:
    return next((a.split("=", 1)[1] for a in argv if a.startswith(f"{name}=")), None)


def _value(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


def session_turn(
    prompt: str, argv: list[str], scenario: dict
) -> tuple[list[dict], int]:
    """The stdout lines (and exit code) of one turn against an on-disk session transcript."""
    resume = _flag(argv, "--resume")
    session_id = resume or _flag(argv, "--session-id")
    assert session_id, argv
    config = Path(os.environ["CLAUDE_CONFIG_DIR"])
    path = (
        config
        / "projects"
        / project_key_for_directory(os.getcwd())
        / f"{session_id}.jsonl"
    )

    def result(
        text: str, *, error: bool = False, structured_output: object = None
    ) -> dict[str, object]:
        message: dict[str, object] = {
            "type": "result",
            "subtype": "error_during_execution" if error else "success",
            "duration_ms": 10,
            "duration_api_ms": 9,
            "is_error": error,
            "num_turns": 1,
            "session_id": session_id,
            "result": text,
            "errors": [text] if error else [],
        }
        if structured_output is not None:
            message["structured_output"] = structured_output
        return message

    if resume and not path.exists():
        return [
            result(f"No conversation found with session ID: {session_id}", error=True)
        ], 1
    prior = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    turn_n = sum(e["type"] == "user" for e in prior) + 1
    reply = f"turn {turn_n}: {prompt}"
    outputs = scenario.get("structured_outputs")
    structured = (
        outputs[turn_n - 1]
        if isinstance(outputs, list) and 0 < turn_n <= len(outputs)
        else None
    )
    entries = [
        {
            "type": "user",
            "uuid": str(uuid.uuid4()),
            "sessionId": session_id,
            "message": {"role": "user", "content": prompt},
        },
        {
            "type": "assistant",
            "uuid": str(uuid.uuid4()),
            "sessionId": session_id,
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": reply}],
            },
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.writelines(json.dumps(e) + "\n" for e in entries)
    lines: list[dict] = []
    if "--session-mirror" in argv:
        lines.append(
            {"type": "transcript_mirror", "filePath": str(path), "entries": entries}
        )
    lines.append(
        {
            "type": "assistant",
            "message": {
                "id": f"msg_{uuid.uuid4().hex}",
                "model": "m",
                "usage": {},
                "content": [{"type": "text", "text": reply}],
            },
            "parent_tool_use_id": None,
            "session_id": session_id,
            "uuid": str(uuid.uuid4()),
        }
    )
    lines.append(result(reply, structured_output=structured))
    return lines, 0


INTERRUPTED = (
    "[Tool call interrupted: the session ended before this call's result was recorded, so its "
    "outcome is unknown. Check whether it took effect before relying on it or running it again.]"
)
NO_RESPONSE = "No response requested."
NOTHING_PENDING = "Nothing pending. What should I continue with?"


class Transcript:
    """One session's transcript the way Claude Code 2.1.294 keeps it: each whole entry goes to
    the local file at once, and `--session-mirror` frames carry it to the SDK in groups (the
    prompt, then one group per model round). The newest ``lag`` groups are held back, so the
    store trails the stream: a crash loses the round in flight even though its tool ran.
    """

    def __init__(
        self,
        path: Path,
        session_id: str,
        emit,  # type: ignore
        *,
        mirror: bool,
        lag: int,  # type: ignore
    ) -> None:
        """Load the transcript file and hold back the newest ``lag`` groups."""
        self.path, self.session_id = path, session_id
        self.entries = (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )
        self._emit, self._mirror, self._lag = emit, mirror, lag
        self._groups: list[list[dict]] = [[]]

    def add(self, entry: dict) -> dict:
        """Append one entry to the file and the current mirror group."""
        entry = {
            "uuid": str(uuid.uuid4()),
            "sessionId": self.session_id,
            "parentUuid": self.entries[-1]["uuid"] if self.entries else None,
            **entry,
        }
        self.entries.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        self._groups[-1].append(entry)
        if self._lag == 0:
            self.mirror()
        return entry

    def next_round(self) -> None:
        """Start a mirror group and flush groups older than ``lag``."""
        self._groups.append([])
        self.mirror(keep=self._lag)

    def mirror(self, keep: int = 0) -> None:
        """Emit stored groups, holding back the newest ``keep``."""
        while len(self._groups) > keep:
            group = self._groups.pop(0)
            if group and self._mirror:
                self._emit(
                    {
                        "type": "transcript_mirror",
                        "filePath": str(self.path),
                        "entries": group,
                    }
                )
        self._groups = self._groups or [[]]

    def repair(self) -> list[str]:
        """What the real CLI appends on `--resume` before the new prompt: an "outcome unknown"
        error result for every tool call with no recorded result, then a synthetic "No response
        requested." when the last turn did not end on Claude's text. Returns what it added.
        """
        added = []
        results = {
            b["tool_use_id"]
            for e in self.entries
            for b in content_blocks(e)
            if b.get("type") == "tool_result"
        }
        for call in [
            b
            for e in self.entries
            for b in content_blocks(e)
            if b.get("type") == "tool_use"
        ]:
            if call["id"] not in results:
                self.add(
                    {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": call["id"],
                                    "content": INTERRUPTED,
                                    "is_error": True,
                                }
                            ],
                        },
                    }
                )
                added.append(f"interrupted:{call['id']}")
        last = self.entries[-1] if self.entries else None
        if last and not (
            last["type"] == "assistant"
            and all(b.get("type") == "text" for b in content_blocks(last))
        ):
            self.add(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "model": "<synthetic>",
                        "content": [{"type": "text", "text": NO_RESPONSE}],
                    },
                }
            )
            added.append("no_response")
        return added


def _is_prompt(entry: dict) -> bool:
    return entry["type"] == "user" and isinstance(
        entry.get("message", {}).get("content"), str
    )


def plan_steps(
    transcript: Transcript, prompt: str, scenario: dict, resumed: dict | None = None
) -> list[dict]:
    """The steps Claude still has to take for this prompt (the prompt's entry is already added).

    A new prompt starts its plan from the top — re-sending a prompt the store already holds
    repeats every tool call, as the real model usually does. A continuation prompt
    (one starting "Continue") carries on the last task prompt's plan (across earlier
    continuations) when the
    transcript shows that turn was cut off (it ends in the synthetic "No response requested."),
    skipping each tool step whose call has a real (not "interrupted") result; otherwise there is
    nothing to continue and Claude says so, losing a prompt that never reached the store.
    ``resumed`` is the deferred call this run executes before the model continues; it counts as
    done.
    """
    *prior, current = transcript.entries  # type: ignore
    start = len(prior)
    if prompt.startswith("Continue"):
        last = prior[-1] if prior else {}
        cut_off = last.get("isApiErrorMessage") or (
            content_blocks(last) and content_blocks(last)[0].get("text") == NO_RESPONSE
        )
        if not cut_off:
            return [{"text": NOTHING_PENDING}]
        start = max(
            i
            for i, e in enumerate(prior)
            if _is_prompt(e) and not e["message"]["content"].startswith("Continue")
        )
        prompt = prior[start]["message"]["content"]
    steps = next(
        (s for prefix, s in scenario["plans"] if prompt.startswith(prefix)),
        [{"text": f"reply: {prompt}"}],
    )
    calls = {
        b["id"]: b
        for e in transcript.entries[start:]
        for b in content_blocks(e)
        if b.get("type") == "tool_use"
    }
    done = [
        (calls[b["tool_use_id"]]["name"], calls[b["tool_use_id"]]["input"])
        for e in transcript.entries[start:]
        for b in content_blocks(e)
        if b.get("type") == "tool_result"
        and b.get("content") != INTERRUPTED
        and b["tool_use_id"] in calls
    ]
    if resumed is not None:
        done.append((resumed["name"], resumed.get("input", {})))
    for k, step in enumerate(steps):
        key = (
            (f"mcp__{SERVER}__{step['tool']}", step.get("input", {}))
            if "tool" in step
            else (step.get("native"), step.get("input", {}))
        )
        if "text" in step or key not in done:
            return steps[k:]
        done.remove(key)
    return []


def plan_lines(steps: list[dict], session_id: str) -> list[dict]:
    """Stream lines (and fake directives) for ``steps``: one model round per tool step, with a
    new tool_use_id every time, and the final text streamed in chunks.
    """

    def ev(event: dict) -> dict:
        return {
            "type": "stream_event",
            "event": event,
            "session_id": session_id,
            "parent_tool_use_id": None,
            "uuid": str(uuid.uuid4()),
        }

    def round_start(message_id: str) -> list[dict]:
        return [
            {"fake": "round"},
            ev(
                {
                    "type": "message_start",
                    "message": {
                        "model": "m",
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "usage": {"input_tokens": 9, "output_tokens": 1},
                    },
                }
            ),
        ]

    def round_end(stop_reason: str) -> list[dict]:
        return [
            ev(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason},
                    "usage": {"input_tokens": 9, "output_tokens": 5},
                }
            ),
            ev({"type": "message_stop"}),
        ]

    def assistant(message_id: str, block: dict) -> dict:
        return {
            "type": "assistant",
            "message": {
                "model": "m",
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [block],
                "usage": {},
            },
            "parent_tool_use_id": None,
            "session_id": session_id,
            "uuid": str(uuid.uuid4()),
        }

    lines: list[dict] = []
    for step in steps:
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        if "text" in step:
            words = step["text"].split(" ")
            chunks = [w + " " for w in words[:-1]] + words[-1:]
            deltas = [
                ev(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": c},
                    }
                )
                for c in chunks
            ]
            half = max(1, len(deltas) // 2)
            lines += [
                *round_start(message_id),
                ev(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    }
                ),
                *deltas[:half],
                {"fake": "trigger", "point": "mid_answer"},
                *deltas[half:],
                assistant(message_id, {"type": "text", "text": step["text"]}),
                ev({"type": "content_block_stop", "index": 0}),
                *round_end("end_turn"),
                {
                    "fake": "plan_result",
                    "text": step["text"],
                    **(
                        {"structured_output": step["structured_output"]}
                        if "structured_output" in step
                        else {}
                    ),
                },
            ]
            break
        tool_use_id = f"toolu_{uuid.uuid4().hex[:12]}"
        tool_input = step.get("input", {})
        name = f"mcp__{SERVER}__{step['tool']}" if "tool" in step else step["native"]
        block = {
            "type": "tool_use",
            "id": tool_use_id,
            "name": name,
            "input": tool_input,
        }
        lines += [
            *round_start(message_id),
            ev(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {**block, "input": {}},
                }
            ),
            ev(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(tool_input),
                    },
                }
            ),
            assistant(message_id, block),
        ]
        if "tool" in step:
            lines += [
                {
                    "fake": "mcp_calls",
                    "calls": [
                        {
                            "id": tool_use_id,
                            "name": step["tool"],
                            "arguments": tool_input,
                        }
                    ],
                },
                ev({"type": "content_block_stop", "index": 0}),
                *round_end("tool_use"),
            ]
        else:
            lines += [
                ev({"type": "content_block_stop", "index": 0}),
                *round_end("tool_use"),
                {
                    "fake": "native_call",
                    "id": tool_use_id,
                    "name": name,
                    "input": tool_input,
                    **{k: v for k, v in step.items() if k not in ("native", "input")},
                },
            ]
    return lines


def main() -> int:
    """Run the scripted CLI from the scenario file in ``argv``."""
    with open(sys.argv[1]) as f:
        scenario = json.load(f)
    argv = sys.argv[2:]
    if "-v" in argv or "--version" in argv:
        print(f"{scenario.get('version', '2.1.294')} (Claude Code)")
        return 0
    child = (
        subprocess.Popen(
            scenario["child"], start_new_session=bool(scenario.get("detach_child"))
        )
        if scenario.get("child")
        else None
    )
    if scenario.get("record"):
        tmp = f"{scenario['record']}.tmp"
        with open(tmp, "w") as f:
            json.dump(
                {"pid": os.getpid(), "argv": argv, "child_pid": child and child.pid}, f
            )
        os.replace(tmp, scenario["record"])
    if scenario.get("ignore_sigterm"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if scenario.get("stderr"):
        sys.stderr.write(scenario["stderr"])
        sys.stderr.flush()

    def emit(obj: dict) -> None:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    state: dict = {"interrupted": False, "mcp_ready": False, "next_id": 0, "hooks": {}}
    session_id = _flag(argv, "--resume") or _flag(argv, "--session-id")

    def answer_control(msg: dict) -> None:
        emit(
            {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": msg["request_id"],
                    "response": {},
                },
            }
        )
        if msg["request"].get("subtype") == "initialize":
            state["hooks"] = msg["request"].get("hooks") or {}
        if msg["request"].get("subtype") == "interrupt":
            state["interrupted"] = True
            if scenario.get("interrupt_result"):
                emit(scenario["interrupt_result"])

    def listen_for_control() -> None:
        """Answer a control request (an interrupt) that arrived while the turn streams, as the
        real CLI does; single-threaded, so only between two scripted lines.
        """
        while select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if not line:
                return
            msg = json.loads(line)
            if msg.get("type") == "control_request":
                answer_control(msg)

    def mcp_send(messages: list[dict]) -> set[str]:
        """Send JSON-RPC messages to the SDK MCP server; the request ids awaiting a response."""
        waiting = set()
        for message in messages:
            state["next_id"] += 1
            request_id = f"mcp_{state['next_id']}"
            if "id" in message:
                waiting.add(request_id)
            emit(
                {
                    "type": "control_request",
                    "request_id": request_id,
                    "request": {
                        "subtype": "mcp_message",
                        "server_name": SERVER,
                        "message": message,
                    },
                }
            )
        return waiting

    def mcp(messages: list[dict]) -> dict[str, dict]:
        """Send JSON-RPC messages to the SDK MCP server; return responses by request id."""
        return mcp_wait(mcp_send(messages))

    def mcp_wait(waiting: set[str]) -> dict[str, dict]:
        responses: dict[str, dict] = {}
        if not waiting:
            return responses
        for line in sys.stdin:
            msg = json.loads(line)
            if msg.get("type") == "control_request":
                answer_control(msg)
            elif msg.get("type") == "control_response":
                response = msg["response"]
                if response.get("request_id") in waiting:
                    responses[response["request_id"]] = response
                    waiting.discard(response["request_id"])
            if not waiting:
                break
        return responses

    def control(request: dict) -> dict:
        """One control request to the SDK; its response (answering the SDK's own meanwhile)."""
        state["next_id"] += 1
        request_id = f"ctl_{state['next_id']}"
        emit({"type": "control_request", "request_id": request_id, "request": request})
        for line in sys.stdin:
            msg = json.loads(line)
            if msg.get("type") == "control_request":
                answer_control(msg)
            elif (
                msg.get("type") == "control_response"
                and msg["response"].get("request_id") == request_id
            ):
                return msg["response"]
        return {"subtype": "error", "error": "stdin closed"}

    def hook_input(event: str, call: dict, **extra: object) -> dict:
        return {
            "hook_event_name": event,
            "session_id": session_id,
            "transcript_path": "",
            "cwd": os.getcwd(),
            "tool_name": call["name"],
            "tool_input": call.get("input", {}),
            "tool_use_id": call["id"],
            **extra,
        }

    def run_command_hooks(
        event: str, call: dict, **extra: object
    ) -> list[tuple[str | None, str | None]]:
        """Run every matching `--settings` command hook; each one's (decision, reason)."""
        settings = json.loads(_value(argv, "--settings") or "{}")
        verdicts: list[tuple[str | None, str | None]] = []
        for matcher in (settings.get("hooks") or {}).get(event) or []:
            pattern = matcher.get("matcher")
            if pattern not in (None, "", "*") and call["name"] not in pattern.split(
                "|"
            ):
                continue
            for hook in matcher.get("hooks", []):
                try:
                    proc = subprocess.run(
                        hook["command"],
                        shell=True,
                        input=json.dumps(hook_input(event, call, **extra)),
                        capture_output=True,
                        text=True,
                        timeout=hook.get("timeout", 60),
                    )
                except subprocess.TimeoutExpired:
                    verdicts.append(("error", "timeout"))
                    continue
                if proc.returncode == 2:
                    verdicts.append(("deny", proc.stderr.strip()))
                elif proc.returncode != 0:
                    verdicts.append(("error", f"exit {proc.returncode}"))
                elif not proc.stdout.strip():
                    verdicts.append((None, None))
                else:
                    try:
                        output = json.loads(proc.stdout)["hookSpecificOutput"]
                        verdicts.append(
                            (
                                output["permissionDecision"],
                                output.get("permissionDecisionReason"),
                            )
                        )
                    except (ValueError, KeyError, TypeError):
                        verdicts.append(("error", "invalid output"))
        return verdicts

    def run_hooks(event: str, call: dict, **extra: object) -> list[str | None]:
        """Run every matching SDK hook callback for ``event``; each one's decision."""
        decisions: list[str | None] = []
        for matcher in state["hooks"].get(event) or []:
            pattern = matcher.get("matcher")
            if pattern not in (None, "", "*") and call["name"] not in pattern.split(
                "|"
            ):
                continue
            for callback_id in matcher.get("hookCallbackIds", []):
                response = control(
                    {
                        "subtype": "hook_callback",
                        "callback_id": callback_id,
                        "tool_use_id": call["id"],
                        "input": hook_input(event, call, **extra),
                    }
                )
                if response.get("subtype") == "error":
                    decisions.append("error")
                else:
                    output = (response.get("response") or {}).get(
                        "hookSpecificOutput"
                    ) or {}
                    decisions.append(output.get("permissionDecision"))
        return decisions

    def pre_tool_use(call: dict) -> str | None:
        verdicts = run_command_hooks("PreToolUse", call)
        decisions = [d for d, _ in verdicts] + run_hooks("PreToolUse", call)
        state["entry"]["hooks"].append(
            {
                "id": call["id"],
                "name": call["name"],
                "decisions": decisions,
                "reasons": [r for _, r in verdicts],
            }
        )
        for decision in ("deny", "defer", "allow"):
            if decision in decisions:
                return decision
        return None

    def permitted(call: dict) -> bool:
        allowed = (
            argv[argv.index("--allowedTools") + 1] if "--allowedTools" in argv else ""
        )
        if call["name"] in allowed.split(","):
            return True
        if "--permission-prompt-tool" in argv:
            response = control(
                {
                    "subtype": "can_use_tool",
                    "tool_name": call["name"],
                    "input": call.get("input", {}),
                    "tool_use_id": call["id"],
                    "permission_suggestions": [],
                }
            )
            behavior = (response.get("response") or {}).get("behavior")
            state["entry"]["permission"].append(
                {"id": call["id"], "behavior": behavior}
            )
            return behavior == "allow"
        return False

    def tool_result(call: dict, text: str, *, error: bool) -> None:
        message = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": text,
                    "is_error": error,
                }
            ],
        }
        if state.get("transcript"):
            state["transcript"].add({"type": "user", "message": message})
        emit(
            {
                "type": "user",
                "message": message,
                "parent_tool_use_id": None,
                "session_id": session_id,
                "uuid": f"tr-{call['id']}-{uuid.uuid4().hex[:6]}",
            }
        )

    def trigger(point: str) -> None:
        """At the scenario's ``crash_at`` point (its first occurrence in this process, in a
        turn whose prompt starts with ``crash_prompt`` when set, and only while
        ``<trigger_dir>/<point>`` does not exist yet): wait ``trigger_settle`` seconds so the SDK
        stores the frames already mirrored, write that marker (pid, session), and wait to be
        killed, or with ``crash_mode: "die"`` SIGKILL this CLI itself.
        """
        if scenario.get("crash_at") != point:
            return
        if not state["entry"]["prompt"].startswith(scenario.get("crash_prompt", "")):
            return
        state["hits"] = state.get("hits", 0) + 1
        marker = Path(scenario.get("trigger_dir") or ".") / point
        if state["hits"] != 1 or marker.exists():
            return
        time.sleep(scenario.get("trigger_settle", 0.3))
        marker.parent.mkdir(parents=True, exist_ok=True)
        tmp = marker.with_name(f".{point}.tmp")
        tmp.write_text(
            json.dumps({"point": point, "pid": os.getpid(), "session_id": session_id})
        )
        os.replace(tmp, marker)
        if scenario.get("crash_mode") == "api_error":
            api_error()
        if scenario.get("crash_mode") == "die" and sys.platform != "win32":
            os.kill(os.getpid(), signal.SIGKILL)
        time.sleep(scenario.get("trigger_block", 60))
        os._exit(70)

    def api_error() -> None:
        """End the run the way the CLI ends one whose API calls kept failing after its own
        retries: the synthetic error reply in the transcript (mirrored with the rest), the error
        result, exit 1.
        """
        status = scenario.get("api_error_status", 529)
        text = f"API Error: {status} " + ("overloaded" if status == 529 else "error")
        if state.get("transcript"):
            state["transcript"].add(
                {
                    "type": "assistant",
                    "isApiErrorMessage": True,
                    "message": {
                        "role": "assistant",
                        "model": "<synthetic>",
                        "content": [{"type": "text", "text": text}],
                    },
                }
            )
            state["transcript"].mirror()
        emit(
            {
                "type": "result",
                "subtype": "success",
                "duration_ms": 10,
                "duration_api_ms": 9,
                "is_error": True,
                "num_turns": state.get("rounds", 0),
                "session_id": session_id,
                "result": text,
                "errors": [],
                "api_error_status": status,
                "terminal_reason": "api_error",
                "total_cost_usd": 0.0,
            }
        )
        os._exit(1)

    def deferred_store() -> dict:
        path = Path(scenario["state"])
        return json.loads(path.read_text()) if path.exists() else {}

    def save_deferred(call: dict | None) -> None:
        store = deferred_store()
        if call is None:
            store.pop(session_id, None)
        else:
            store[session_id] = call
        Path(scenario["state"]).write_text(json.dumps(store))

    def stale_deferral_result() -> dict:
        """What 2.1.294 ends a resumed run with when the deferred call's tool is not in --tools:
        nothing runs, the prompt is not handled, and the marker stays.
        """
        return {
            "type": "result",
            "subtype": "error_during_execution",
            "duration_ms": 5,
            "duration_api_ms": 0,
            "is_error": True,
            "num_turns": 0,
            "session_id": session_id,
            "stop_reason": None,
            "terminal_reason": "tool_deferred_unavailable",
            "errors": [],
        }

    def mirror_scripted_prompt() -> None:
        """A scripted deferral has no Transcript. Mirror the prompt when a SessionStore is
        attached, so the next resume is not rejected for an empty store.
        """
        if "--session-mirror" not in argv or not session_id:
            return
        config = os.environ.get("CLAUDE_CONFIG_DIR")
        if not config:
            return
        path = (
            Path(config)
            / "projects"
            / project_key_for_directory(os.getcwd())
            / f"{session_id}.jsonl"
        )
        emit(
            {
                "type": "transcript_mirror",
                "filePath": str(path),
                "entries": [
                    {
                        "type": "user",
                        "uuid": str(uuid.uuid4()),
                        "sessionId": session_id,
                        "message": {
                            "role": "user",
                            "content": state["entry"]["prompt"],
                        },
                    }
                ],
            }
        )

    def native_call(call: dict) -> bool:
        """Gate and maybe run one native call; True when it deferred (the turn ends)."""
        trigger("after_tool_call")
        decision = pre_tool_use(call)
        if decision == "defer" and call.get("batch", 1) == 1:
            save_deferred(call)
            if state.get("transcript"):
                state["transcript"].mirror()
            else:
                mirror_scripted_prompt()
            emit(
                {
                    "type": "result",
                    "subtype": "success",
                    "duration_ms": 5,
                    "duration_api_ms": 4,
                    "is_error": False,
                    "num_turns": 1,
                    "session_id": session_id,
                    "stop_reason": "tool_deferred",
                    "result": "",
                    "deferred_tool_use": {
                        "id": call["id"],
                        "name": call["name"],
                        "input": call.get("input", {}),
                    },
                    **call.get("deferred_result", {}),
                }
            )
            return True
        # A deferral the CLI ignored (batch) or no decision at all falls to the permission layer,
        # where 2.1.294 approves reads itself. An `ask` rule outranks both that and a hook's allow.
        settings = json.loads(_value(argv, "--settings") or "{}")
        asked = call["name"] in (settings.get("permissions") or {}).get("ask", [])
        if decision == "deny":
            runs = False
        elif not asked and (decision == "allow" or call["name"] in _AUTO_READS):
            runs = True
        else:
            runs = permitted(call)
        if runs:
            for name, text in (call.get("writes") or {}).items():
                Path(name).write_text(
                    text
                )  # relative to the cwd: the session's workspace
            with open(scenario["effects"], "a") as f:
                f.write(
                    json.dumps(
                        {
                            "id": call["id"],
                            "name": call["name"],
                            "input": call.get("input", {}),
                            "pid": os.getpid(),
                        }
                    )
                    + "\n"
                )
            trigger("after_tool_run")
            run_command_hooks(
                "PostToolUse", call, tool_response={"stdout": "ok", "stderr": ""}
            )
            run_hooks("PostToolUse", call, tool_response={"stdout": "ok", "stderr": ""})
            tool_result(call, "ok", error=False)
        else:
            tool_result(call, "denied", error=True)
        return False

    in_flight: list[
        tuple[dict, str]
    ] = []  # (call, MCP request id), answered by await_mcp_calls

    def mcp_calls(calls: list[dict]) -> None:
        trigger("after_tool_call")
        results = state.setdefault("results", {})
        raw = json.dumps(calls)
        for tool_use_id, text in results.items():
            raw = raw.replace("{{RESULT:%s}}" % tool_use_id, text)
        calls = json.loads(raw)
        for call in calls:
            pre_tool_use(
                {
                    "id": call["id"],
                    "name": f"mcp__{SERVER}__{call['name']}",
                    "input": call.get("arguments", {}),
                }
            )
        if not state["mcp_ready"]:
            mcp(
                [
                    {
                        "jsonrpc": "2.0",
                        "id": "init",
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {},
                            "clientInfo": {"name": "fake-claude", "version": "0"},
                        },
                    },
                    {"jsonrpc": "2.0", "method": "notifications/initialized"},
                ]
            )
            state["mcp_ready"] = True
        for call in calls:
            [request_id] = mcp_send(
                [
                    {
                        "jsonrpc": "2.0",
                        "id": call["id"],
                        "method": "tools/call",
                        "params": {
                            "name": call["name"],
                            "arguments": call.get("arguments", {}),
                            "_meta": {"claudecode/toolUseId": call["id"]},
                        },
                    }
                ]
            )
            in_flight.append((call, request_id))

    def await_mcp_calls() -> None:
        """Wait for every in-flight tool call's response and emit its tool result."""
        if not in_flight:
            return
        responses = mcp_wait({request_id for _, request_id in in_flight})
        for call, request_id in in_flight:
            response = responses.get(request_id, {})
            result = (
                (response.get("response") or {}).get("mcp_response", {}).get("result")
            )
            state["entry"]["mcp"].append(
                {"id": call["id"], "result": result, "error": response.get("error")}
            )
            content = (result or {}).get("content") or []
            text = "".join(c.get("text", "") for c in content) or str(
                response.get("error")
            )
            state.setdefault("results", {})[call["id"]] = text
            tool_result(call, text, error=result is None or bool(result.get("isError")))
        in_flight.clear()
        trigger("after_tool_run")

    def ends_message(out: dict) -> bool:
        """True for a line after the model message whose tool calls are in flight."""
        if out.get("type") == "stream_event":
            return out["event"].get("type") == "message_start"
        return out.get("type") in ("user", "result", "system") or out.get("fake") in (
            "native_call",
            "round",
            "plan_result",
        )

    def plan_turn(
        prompt: str, prompt_uuid: str | None
    ) -> tuple[list[dict], int | None]:
        """The crash/resume model's turn for one prompt (see "plans" above)."""
        config = Path(os.environ["CLAUDE_CONFIG_DIR"])
        path = (
            config
            / "projects"
            / project_key_for_directory(os.getcwd())
            / f"{session_id}.jsonl"
        )
        resume = _flag(argv, "--resume")
        if resume and not path.exists():
            return [
                session_error(f"No conversation found with session ID: {session_id}")
            ], 1
        transcript = state.get("transcript")
        if transcript is None:
            if not resume and path.exists():
                return [session_error(f"Session ID {session_id} is already in use.")], 1
            transcript = state["transcript"] = Transcript(
                path,
                session_id,  # type: ignore
                emit,
                mirror="--session-mirror" in argv,
                lag=scenario.get("mirror_lag_rounds", 1),
            )
            if resume:
                state["entry"]["repairs"] = transcript.repair()
        transcript.add(
            {
                "type": "user",
                "message": {"role": "user", "content": prompt},
                **({"uuid": prompt_uuid} if prompt_uuid else {}),
            }
        )
        trigger("before_prompt_persisted")
        resumed = (
            deferred_store().get(session_id) if resume and "state" in scenario else None
        )
        steps = plan_steps(transcript, prompt, scenario, resumed)
        state["entry"]["steps"] = steps
        return plan_lines(steps, session_id), None  # type: ignore

    def session_error(text: str) -> dict:
        return {
            "type": "result",
            "subtype": "error_during_execution",
            "duration_ms": 10,
            "duration_api_ms": 9,
            "is_error": True,
            "num_turns": 0,
            "session_id": session_id,
            "result": text,
            "errors": [text],
        }

    def plan_result(text: str, structured_output: object = None) -> None:
        state["transcript"].mirror()
        message: dict[str, object] = {
            "type": "result",
            "subtype": "success",
            "duration_ms": 10,
            "duration_api_ms": 9,
            "is_error": False,
            "num_turns": 1,
            "session_id": session_id,
            "result": text,
            "total_cost_usd": 0.0,
        }
        if structured_output is not None:
            message["structured_output"] = structured_output
        emit(message)

    def answered(tool_use_id: str) -> bool:
        """Whether this run's transcript (``plans`` scenarios keep one) answers the call with
        more than this run's own interrupted patch.
        """
        transcript = state.get("transcript")
        return transcript is not None and any(
            b.get("type") == "tool_result"
            and b.get("tool_use_id") == tool_use_id
            and b.get("content") != INTERRUPTED
            for e in transcript.entries
            for b in content_blocks(e)
        )

    def approved_deferral() -> bool:
        """Whether this run resumes an approved call no earlier run started.

        Read from the hook's state directory, the hook command's last argument.
        """
        try:
            hooks = json.loads(_value(argv, "--settings") or "{}")["hooks"][
                "PreToolUse"
            ]
            state = Path(shlex.split(hooks[0]["hooks"][0]["command"])[-1])
            resumed = json.loads((state / "turn.json").read_text())["resumed"]
            decision = json.loads((state / "decisions" / f"{resumed}.json").read_text())
        except (TypeError, KeyError, IndexError, ValueError, OSError):
            return False
        try:
            calls = json.loads((state / "calls.json").read_text())["calls"]
        except (TypeError, KeyError, ValueError, OSError):
            calls = []  # the real CLI never reads the ledger
        return decision.get("decision") == "approved" and all(
            c.get("tool_use_id") != resumed for c in calls
        )

    def inputs() -> Iterator[dict[str, Any]]:
        """Stdin, plus the turn 2.1.294 starts on its own when it resumes an approved call.

        It starts once initialized; a prompt sent as well is a second turn.
        """
        started = False
        for line in sys.stdin:
            msg = json.loads(line)
            yield msg
            initialized = (
                msg.get("type") == "control_request"
                and msg["request"].get("subtype") == "initialize"
            )
            if initialized and not started and "--resume" in " ".join(argv):
                started = True
                if approved_deferral():
                    yield {
                        "type": "user",
                        "message": {"role": "user", "content": "Continue."},
                    }

    for msg in inputs():
        if msg.get("type") == "control_request":
            answer_control(msg)
        elif msg.get("type") == "user":
            prompt = msg["message"]["content"]
            if isinstance(prompt, list):
                prompt = "".join(
                    b.get("text", "") for b in prompt if b.get("type") == "text"
                )
            entry = {
                "pid": os.getpid(),
                "argv": argv,
                "cwd": os.getcwd(),
                "config_dir": os.environ.get("CLAUDE_CONFIG_DIR"),
                "start": time.time(),
                "prompt": prompt,
                "hooks": [],
                "permission": [],
            }
            state["entry"] = entry
            exit_code = scenario.get("exit_code")
            if scenario.get("session_mode"):
                turn, exit_code = session_turn(prompt, argv, scenario)
            elif "plans" in scenario:
                turn, exit_code = plan_turn(prompt, msg.get("uuid"))
            else:
                turn = next(
                    (
                        t
                        for prefix, t in scenario.get("turns", [])
                        if prompt.startswith(prefix)
                    ),
                    scenario.get("turn", []),
                )
                turn = json.loads(json.dumps(turn).replace("{{P}}", prompt))
                if scenario.get("state"):
                    store = deferred_store()
                    store["runs"] = store.get("runs", 0) + 1
                    Path(scenario["state"]).write_text(json.dumps(store))
                    turn = json.loads(
                        json.dumps(turn).replace("{{RUN}}", str(store["runs"]))
                    )
            entry["mcp"] = []
            deferred = False
            if scenario.get("state") and "--resume" in " ".join(argv):
                pending = deferred_store().get(session_id)
                if pending is not None and answered(pending["id"]):
                    # The real CLI drops an answered call's deferral marker on load.
                    save_deferred(None)
                    pending = None
                if pending is not None and pending["name"] not in (
                    _value(argv, "--tools") or ""
                ).split(","):
                    emit(stale_deferral_result())
                    deferred = True
                elif pending is not None:
                    save_deferred(None)
                    deferred = native_call(pending)
            for out in [] if deferred else turn:
                if state["interrupted"]:
                    break
                if ends_message(out):
                    await_mcp_calls()
                if out.get("fake") == "mcp_calls":
                    mcp_calls(out["calls"])
                elif out.get("fake") == "native_call":
                    if native_call(out):
                        break
                elif out.get("fake") == "sleep":
                    time.sleep(out["seconds"])
                    if scenario.get("listen_while_streaming"):
                        listen_for_control()
                elif out.get("fake") == "round":
                    state["rounds"] = state.get("rounds", 0) + 1
                    state["transcript"].next_round()
                elif out.get("fake") == "trigger":
                    trigger(out["point"])
                elif out.get("fake") == "plan_result":
                    plan_result(out["text"], out.get("structured_output"))
                else:
                    if out.get("type") == "assistant" and state.get("transcript"):
                        state["transcript"].add(
                            {"type": "assistant", "message": out["message"]}
                        )
                    emit(out)
            await_mcp_calls()
            entry["end"] = time.time()
            if scenario.get("log"):
                with open(scenario["log"], "a") as f:
                    f.write(json.dumps(entry) + "\n")
            if state["interrupted"] or scenario.get("hang_after_prompt"):
                # The prompt's messages are flushed. Tests wait on this before cancelling,
                # so the cancel lands in the turn and not during connect.
                if scenario.get("record"):
                    Path(str(scenario["record"]) + ".ready").touch()
                continue
            if exit_code:
                return int(exit_code)
    if scenario.get("hang_on_eof"):
        while True:
            time.sleep(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
