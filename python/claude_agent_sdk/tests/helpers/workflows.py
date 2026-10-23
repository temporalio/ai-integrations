"""A minimal workflow that runs Claude turns.

Its own module so the worker loads it through the default (sandboxed) workflow runner.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import timedelta
from typing import Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel

    from temporalio.claude_agent_sdk import (
        ClaudeDeferredTool,
        ClaudeSession,
        ClaudeSessionState,
        ClaudeTurnOptions,
        ClaudeTurnResult,
        run_claude_turn,
    )
    from temporalio.claude_agent_sdk._models import (
        BRIDGE_POLL_UPDATE,
        CLAUDE_TURN_BUDGET_ERROR_TYPE,
        CLAUDE_TURN_TIMEOUT_ERROR_TYPE,
    )
    from temporalio.contrib.workflow_streams import WorkflowStream
    from temporalio.exceptions import ApplicationError


async def get_weather(city: str) -> str:
    """Current weather for a city."""
    return f"sunny in {city}"


@workflow.defn
class ClaudeChat:
    """Every message continues one Claude conversation."""

    def __init__(self) -> None:
        self._claude = ClaudeSession()
        self._weather_calls = 0

    @workflow.run
    async def run(self) -> None:
        await workflow.wait_condition(lambda: False)

    @workflow.update
    async def chat(self, text: str) -> str:
        reply = await run_claude_turn(text, session=self._claude)
        return reply.text or ""

    @workflow.update
    async def chat_with_tool(self, text: str) -> str:
        async def weather(city: str) -> str:
            self._weather_calls += 1
            return await get_weather(city)

        weather.__name__ = "get_weather"
        weather.__doc__ = get_weather.__doc__
        reply = await run_claude_turn(text, session=self._claude, tools=[weather])
        return reply.text or ""

    @workflow.query
    def weather_calls(self) -> int:
        return self._weather_calls


@workflow.defn
class ClaudeNative:
    """One native Bash call, approved or denied by the workflow."""

    def __init__(self) -> None:
        self._claude = ClaudeSession()

    @workflow.run
    async def run(self, approve: bool) -> str:
        async def gate(call: ClaudeDeferredTool) -> bool:
            return approve and call.name == "Bash"

        reply = await run_claude_turn(
            "run",
            session=self._claude,
            native_tools=["Bash"],
            approve=gate,
        )
        return reply.text or ""


@workflow.defn
class ClaudeNativeWithTool:
    """One message that calls a workflow tool on both sides of an approved native call."""

    def __init__(self) -> None:
        self._weather_calls = 0

    @workflow.run
    async def run(self) -> int:
        async def weather(city: str) -> str:
            self._weather_calls += 1
            return await get_weather(city)

        async def allow(_call: ClaudeDeferredTool) -> bool:
            return True

        weather.__name__ = "get_weather"
        weather.__doc__ = get_weather.__doc__
        await run_claude_turn(
            "weather twice", tools=[weather], native_tools=["Bash"], approve=allow
        )
        return self._weather_calls


@workflow.defn
class ClaudeNativeDenied:
    """One native Bash call, denied with the workflow's reason."""

    @workflow.run
    async def run(
        self, reason: str, prompt: str = "run", model: str | None = None
    ) -> ClaudeTurnResult:
        async def gate(_call: ClaudeDeferredTool) -> tuple[bool, str]:
            return False, reason

        return await run_claude_turn(
            prompt,
            options=ClaudeTurnOptions(model=model, max_turns=4) if model else None,
            native_tools=["Bash"],
            approve=gate,
        )


@workflow.defn
class ClaudeStopped:
    """The first message is cancelled on ``stop`` or, with native Bash, once its call awaits
    approval; the second is an ordinary message on the same session. Returns the second reply
    and the session's spend."""

    def __init__(self) -> None:
        self._stop = False

    @workflow.signal
    def stop(self) -> None:
        self._stop = True

    @workflow.run
    async def run(
        self, first: str, second: str, model: str | None = None, native: bool = False
    ) -> dict[str, Any]:
        session = ClaudeSession()
        options = ClaudeTurnOptions(model=model, max_turns=4) if model else None
        native_tools = ["Bash"] if native else []

        async def hold(_call: ClaudeDeferredTool) -> bool:
            self._stop = True
            await workflow.wait_condition(lambda: False)
            return False

        stopped = asyncio.create_task(
            run_claude_turn(
                first,
                session=session,
                options=options,
                native_tools=native_tools,
                approve=hold if native else None,
            )
        )
        await workflow.wait_condition(lambda: self._stop)
        stopped.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stopped
        reply = await run_claude_turn(
            second, session=session, options=options, native_tools=native_tools
        )
        return {"text": reply.text, "cost_usd": session.state.cost_usd}


@workflow.defn
class ClaudeNativeSequence:
    """One session, one message per prompt, every native call approved and recorded."""

    @workflow.run
    async def run(
        self, prompts: list[str], model: str, tools: list[str] | None = None
    ) -> dict[str, list[Any]]:
        session, approved, inputs, denials, texts = ClaudeSession(), [], [], [], []

        async def allow(call: ClaudeDeferredTool) -> bool:
            approved.append(call.tool_use_id)
            inputs.append(call.input)
            return True

        for prompt in prompts:
            reply = await run_claude_turn(
                prompt,
                session=session,
                options=ClaudeTurnOptions(model=model, max_turns=8),
                native_tools=tools or ["Bash"],
                approve=allow,
            )
            texts.append(reply.text or "")
            denials.extend(reply.permission_denials or [])
        return {
            "approved": approved,
            "inputs": inputs,
            "denials": denials,
            "finished": [
                c.tool_use_id
                for c in session.state.native_calls
                if c.status == "finished"
            ],
            "texts": texts,
        }


@workflow.defn
class ClaudeNativeSafety:
    """Native Bash, always approved, with an optional turn cap.

    ``max_turns`` and ``max_budget_usd`` stop after the deferral that uses the cap up and then
    settle that call. Any other failure, including a resume that defers the same call again,
    fails the workflow.
    """

    def __init__(self) -> None:
        self._claude = ClaudeSession()

    @workflow.run
    async def run(self, mode: str) -> str:
        async def allow(_call: ClaudeDeferredTool) -> bool:
            return True

        options = None
        if mode == "max_turns":
            options = ClaudeTurnOptions(max_turns=1)
        elif mode == "max_budget_usd":
            options = ClaudeTurnOptions(max_budget_usd=0.5)
        try:
            reply = await run_claude_turn(
                "run",
                session=self._claude,
                native_tools=["Bash"],
                approve=allow,
                options=options,
            )
        except ApplicationError as exc:
            if exc.type != CLAUDE_TURN_BUDGET_ERROR_TYPE:
                raise
            flag = "non-retryable" if exc.non_retryable else "retryable"
            settled = await run_claude_turn(
                "settle",
                session=self._claude,
                native_tools=["Bash"],
                approve=allow,
            )
            return f"{exc.type}|{flag}|{settled.text or ''}"
        return reply.text or ""


@workflow.defn
class ClaudeNativeTool:
    """One named native tool, approved. Tools that cannot be made durable must fail the run."""

    @workflow.run
    async def run(self, tool: str) -> str:
        async def allow(_call: ClaudeDeferredTool) -> bool:
            return True

        reply = await run_claude_turn("run", native_tools=[tool], approve=allow)
        return reply.text or ""


@workflow.defn
class ClaudeBounded:
    """A turn with a workflow-side timeout."""

    @workflow.run
    async def run(self) -> str:
        reply = await run_claude_turn("hi", timeout=timedelta(seconds=1))
        return reply.text or ""


@workflow.defn
class ClaudeResume:
    """A turn the workflow resumes after a transient API error."""

    @workflow.run
    async def run(self) -> str:
        reply = await run_claude_turn("What's the weather?")
        return reply.text or ""


@workflow.defn
class ClaudeFailover:
    """A native write, then a later turn that restores the committed workspace.

    The later turn keeps the session id and workspace revision and clears ``started``.
    There is no SessionStore, so a resume would ask the CLI for a transcript the next
    disk does not have.
    """

    def __init__(self) -> None:
        self._claude = ClaudeSession()
        self._held = False
        self._go = False

    @workflow.run
    async def run(self) -> str:
        try:
            await run_claude_turn(
                "write",
                session=self._claude,
                native_tools=["Bash"],
                approve=self._approve,
            )
        except ApplicationError:
            pass
        self._held = True
        await workflow.wait_condition(lambda: self._go)
        carry = self._claude.snapshot()
        carry.started = False
        reply = await run_claude_turn("next", session=ClaudeSession.restore(carry))
        return reply.text or ""

    async def _approve(self, call: ClaudeDeferredTool) -> bool:
        return call.name == "Bash"

    @workflow.signal
    def proceed(self) -> None:
        self._go = True

    @workflow.query
    def held(self) -> bool:
        return self._held


@workflow.defn
class ClaudeDurable:
    """One turn whose heartbeat is short, so a lost worker fails the attempt quickly."""

    def __init__(self) -> None:
        self._claude = ClaudeSession(
            activity_config={
                "heartbeat_timeout": timedelta(seconds=5),
                "start_to_close_timeout": timedelta(seconds=45),
            }
        )
        self._weather_calls = 0

    @workflow.run
    async def run(self, max_budget_usd: float | None = None) -> str:
        async def weather(city: str) -> str:
            self._weather_calls += 1
            return f"sunny in {city}"

        weather.__name__ = "get_weather"
        weather.__doc__ = "Current weather for a city."
        options = (
            ClaudeTurnOptions(max_budget_usd=max_budget_usd)
            if max_budget_usd is not None
            else None
        )
        reply = await run_claude_turn(
            "Weather in Paris?",
            session=self._claude,
            tools=[weather],
            options=options,
        )
        return reply.text or ""

    @workflow.query
    def weather_calls(self) -> int:
        return self._weather_calls


@workflow.defn
class ClaudeHeldTool:
    """One weather call that waits for :meth:`release` before it returns.

    The wait is a workflow condition, so a test can lose the CLI while the call is still
    running and see the reissued call attach to this execution.
    """

    def __init__(self) -> None:
        self._claude = ClaudeSession(
            activity_config={
                "heartbeat_timeout": timedelta(seconds=5),
                "start_to_close_timeout": timedelta(seconds=45),
            }
        )
        self._calls = 0
        self._release = False

    @workflow.run
    async def run(self, effect: str) -> str:
        async def weather(city: str) -> str:
            self._calls += 1
            with workflow.unsafe.sandbox_unrestricted():
                with open(effect, "a", encoding="utf-8") as handle:
                    handle.write(f"{city}\n")
            await workflow.wait_condition(lambda: self._release)
            return f"sunny in {city}"

        weather.__name__ = "get_weather"
        weather.__doc__ = "Current weather for a city."
        reply = await run_claude_turn(
            "Weather in Paris?",
            session=self._claude,
            tools=[weather],
        )
        return reply.text or ""

    @workflow.signal
    def release(self) -> None:
        self._release = True

    @workflow.query
    def calls(self) -> int:
        return self._calls

    @workflow.query
    def bridge_tasks(self) -> int:
        """Tool tasks still on this run's bridge, including one waiting on an in-flight call."""
        handler = workflow.get_update_handler(BRIDGE_POLL_UPDATE)
        bridges = getattr(getattr(handler, "__self__", None), "bridges", None)
        if not isinstance(bridges, dict):
            return 0
        pending = 0
        for bridge in bridges.values():
            tasks = getattr(bridge, "_tasks", None)
            if isinstance(tasks, dict):
                pending += len(tasks)
        return pending


@workflow.defn
class ClaudeTimed:
    """One logical turn whose timeout is measured from the start of the message."""

    @workflow.run
    async def run(self, timeout_s: int) -> float:
        started = workflow.now()
        try:
            await run_claude_turn(
                "What's the weather?",
                timeout=timedelta(seconds=timeout_s),
            )
        except ApplicationError as exc:
            if exc.type != CLAUDE_TURN_TIMEOUT_ERROR_TYPE:
                raise
            return (workflow.now() - started).total_seconds()
        raise ApplicationError("the turn completed", non_retryable=True)


@workflow.defn
class ClaudeSpend:
    """One message on a fresh session: how it failed (empty if it did not) and what the
    session was charged for it."""

    @workflow.run
    async def run(self, timeout_s: float | None = None) -> tuple[str, float]:
        session = ClaudeSession()
        failure = ""
        try:
            await run_claude_turn(
                "hello",
                session=session,
                timeout=timedelta(seconds=timeout_s) if timeout_s else None,
            )
        except ApplicationError as exc:
            failure = exc.type or ""
        return failure, session.state.cost_usd


@workflow.defn
class ClaudeLive:
    """One turn that stays open until it is cancelled or the CLI finishes."""

    @workflow.run
    async def run(self, heartbeat_s: int) -> str:
        reply = await run_claude_turn(
            "hi",
            activity_config={
                "heartbeat_timeout": timedelta(seconds=heartbeat_s),
                # Longer than the time a busy machine needs to reach the hang.
                # After a worker kill the heartbeat timeout is the bound.
                "start_to_close_timeout": timedelta(seconds=90),
            },
        )
        return reply.text or ""


@workflow.defn
class ClaudeContinue:
    """One turn, then continue-as-new with only the session snapshot."""

    def __init__(self) -> None:
        self._first = ""
        self._go = False

    @workflow.run
    async def run(self, state: ClaudeSessionState | None = None) -> str:
        session = ClaudeSession.restore(state) if state is not None else ClaudeSession()
        if state is None:
            reply = await run_claude_turn("hello", session=session)
            self._first = reply.text or ""
            await workflow.wait_condition(lambda: self._go)
            workflow.continue_as_new(session.snapshot())
        reply = await run_claude_turn("again", session=session)
        return reply.text or ""

    @workflow.signal
    def proceed(self) -> None:
        self._go = True

    @workflow.query
    def first(self) -> str:
        return self._first


@workflow.defn
class ClaudeCapped:
    """One message per run, continuing as new with the session snapshot for the next prompt.

    Each message asks for a $5 ``max_budget_usd``; the result is the last reply, or the last
    failure's type.
    """

    @workflow.run
    async def run(
        self,
        prompts: list[str],
        max_cost_usd: float | None = None,
        state: ClaudeSessionState | None = None,
    ) -> str:
        session = (
            ClaudeSession.restore(state)
            if state is not None
            else ClaudeSession(max_cost_usd=max_cost_usd)
        )
        try:
            reply = await run_claude_turn(
                prompts[0],
                session=session,
                options=ClaudeTurnOptions(max_budget_usd=5.0),
            )
            result = reply.text or ""
        except ApplicationError as exc:
            result = exc.type or ""
        if len(prompts) > 1:
            workflow.continue_as_new(args=[prompts[1:], None, session.snapshot()])
        return result


class Forecast(BaseModel):
    """The structured reply a ``ClaudeStructured`` turn must validate into."""

    city: str
    temp_f: int


@workflow.defn
class ClaudeStructured:
    """One turn whose ``output_type`` is ``Forecast``.

    ``heartbeat_s`` is the turn Activity's heartbeat timeout, so a test can lose the
    worker quickly. The validated model is the workflow result.
    """

    @workflow.run
    async def run(self, heartbeat_s: int) -> Forecast:
        reply = await run_claude_turn(
            "forecast",
            output_type=Forecast,
            activity_config={"heartbeat_timeout": timedelta(seconds=heartbeat_s)},
        )
        output = reply.output
        assert isinstance(output, Forecast)
        return output


@workflow.defn
class ClaudeStructuredContinue:
    """One typed turn, then continue-as-new with only the session snapshot.

    Both turns pass ``output_type=Forecast``. The continued run receives the snapshot and
    nothing else.
    """

    def __init__(self) -> None:
        self._first: Forecast | None = None
        self._go = False

    @workflow.run
    async def run(self, state: ClaudeSessionState | None = None) -> Forecast:
        session = ClaudeSession.restore(state) if state is not None else ClaudeSession()
        if state is None:
            reply = await run_claude_turn(
                "forecast", session=session, output_type=Forecast
            )
            output = reply.output
            assert isinstance(output, Forecast)
            self._first = output
            await workflow.wait_condition(lambda: self._go)
            workflow.continue_as_new(session.snapshot())
        reply = await run_claude_turn("again", session=session, output_type=Forecast)
        output = reply.output
        assert isinstance(output, Forecast)
        return output

    @workflow.signal
    def proceed(self) -> None:
        self._go = True

    @workflow.query
    def first(self) -> Forecast | None:
        return self._first


@workflow.defn
class ClaudeStream:
    """One turn that publishes partials on a hosted Workflow Stream."""

    def __init__(self) -> None:
        self.stream = WorkflowStream()
        self._claude = ClaudeSession()
        self._done = False

    @workflow.run
    async def run(self, prompt: str) -> str:
        reply = await run_claude_turn(
            prompt, session=self._claude, streaming_topic="claude"
        )
        # Stay open until the test's subscriber finishes its poll. Completing
        # while that update is in flight fails the workflow task.
        await workflow.wait_condition(lambda: self._done)
        return reply.text or ""

    @workflow.signal
    def finish(self) -> None:
        self._done = True


@workflow.defn
class ClaudeStreamSink:
    """One turn whose partials reach only the worker's partial sink, with a context."""

    @workflow.run
    async def run(self, context: dict[str, str]) -> str:
        reply = await run_claude_turn("hi", streaming_context=context)
        return reply.text or ""


@workflow.defn
class ClaudeStreamUnhosted:
    """Sets a streaming topic and hosts no WorkflowStream."""

    def __init__(self) -> None:
        self._claude = ClaudeSession()

    @workflow.run
    async def run(self, prompt: str) -> str:
        reply = await run_claude_turn(
            prompt, session=self._claude, streaming_topic="claude"
        )
        return reply.text or ""


@workflow.defn
class ClaudeStreamDurable:
    """A streamed turn with a short heartbeat, so a lost CLI can be resumed."""

    def __init__(self) -> None:
        self.stream = WorkflowStream()
        self._claude = ClaudeSession(
            activity_config={
                "heartbeat_timeout": timedelta(seconds=5),
                "start_to_close_timeout": timedelta(seconds=45),
            }
        )
        self._done = False

    @workflow.run
    async def run(self, prompt: str) -> str:
        reply = await run_claude_turn(
            prompt,
            session=self._claude,
            streaming_topic="claude",
            streaming_batch_interval=timedelta(milliseconds=20),
        )
        await workflow.wait_condition(lambda: self._done)
        return reply.text or ""

    @workflow.signal
    def finish(self) -> None:
        self._done = True


@workflow.defn
class ClaudeSandboxSession:
    """Two sandboxed turns. The second waits so a test can switch workers."""

    def __init__(self) -> None:
        self._claude = ClaudeSession()
        self._go = False
        self._snapshot = ""

    @workflow.run
    async def run(self) -> str:
        first = await run_claude_turn("hello", session=self._claude)
        self._snapshot = self._claude.state.sandbox_snapshot or ""
        await workflow.wait_condition(lambda: self._go)
        second = await run_claude_turn("again", session=self._claude)
        carried = self._claude.state.sandbox_snapshot or ""
        return f"{self._snapshot}|{carried}|{first.text}|{second.text}"

    @workflow.signal
    def proceed(self) -> None:
        self._go = True

    @workflow.query
    def snapshot(self) -> str:
        return self._snapshot


@workflow.defn
class ClaudeSandboxContinue:
    """One sandboxed turn, then continue-as-new with the session snapshot."""

    def __init__(self) -> None:
        self._go = False
        self._snapshot = ""

    @workflow.run
    async def run(self, state: ClaudeSessionState | None = None) -> str:
        session = ClaudeSession.restore(state) if state is not None else ClaudeSession()
        if state is None:
            await run_claude_turn("hello", session=session)
            self._snapshot = session.state.sandbox_snapshot or ""
            await workflow.wait_condition(lambda: self._go)
            workflow.continue_as_new(session.snapshot())
        await run_claude_turn("again", session=session)
        return session.state.sandbox_snapshot or ""

    @workflow.signal
    def proceed(self) -> None:
        self._go = True

    @workflow.query
    def snapshot(self) -> str:
        return self._snapshot


@workflow.defn
class ClaudeSandboxFailover:
    """A committed sandboxed turn, then a tool turn whose call waits for :meth:`release`.

    The wait holds the second turn open so a test can lose its worker mid-turn. The heartbeat
    is short so the loss is seen quickly.
    """

    def __init__(self) -> None:
        self._claude: ClaudeSession | None = None
        self._snapshot = ""
        self._calls = 0
        self._release = False

    @workflow.run
    async def run(self, session_id: str) -> str:
        self._claude = ClaudeSession(
            session_id,
            activity_config={
                "heartbeat_timeout": timedelta(seconds=5),
                "start_to_close_timeout": timedelta(minutes=3),
            },
        )
        options = ClaudeTurnOptions(model="claude-haiku-5-5", max_turns=4)
        await run_claude_turn(
            "The code word is bluebird. Reply with exactly: noted",
            session=self._claude,
            options=options,
        )
        self._snapshot = self._claude.state.sandbox_snapshot or ""

        async def weather(city: str) -> str:
            self._calls += 1
            await workflow.wait_condition(lambda: self._release)
            return f"sunny in {city}"

        weather.__name__ = "get_weather"
        weather.__doc__ = "Current weather for a city."
        reply = await run_claude_turn(
            "Call get_weather for Paris. Then reply with only the code word I gave you "
            "and the tool's result.",
            session=self._claude,
            options=options,
            tools=[weather],
        )
        return reply.text or ""

    @workflow.signal
    def release(self) -> None:
        self._release = True

    @workflow.query
    def calls(self) -> int:
        return self._calls

    @workflow.query
    def snapshots(self) -> list[str]:
        """The first turn's committed snapshot, then the session's current one."""
        current = self._claude.state.sandbox_snapshot if self._claude else None
        return [self._snapshot, current or ""]


@workflow.defn
class ClaudeSandboxLive:
    """A sandboxed turn bounded by a workflow timeout, so a test can cancel or time it out."""

    @workflow.run
    async def run(self, timeout_s: int) -> str:
        reply = await run_claude_turn("hi", timeout=timedelta(seconds=timeout_s))
        return reply.text or ""
