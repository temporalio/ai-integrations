"""Workflows shared by the bundled-runtime and replay tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import timedelta

from openai_codex import (
    AsyncCodex,
    AsyncThread,
    CodexConfig,
    InvalidParamsError,
    Sandbox,
    TransportClosedError,
    TurnResult,
)
from openai_codex.generated.v2_all import ThreadStartResponse
from openai_codex.models import JsonObject

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.openai_codex import TemporalCodexClient

with workflow.unsafe.imports_passed_through():
    from tests.helpers.protocol import Echo


@workflow.defn
class ChatWorkflow:
    @workflow.run
    async def run(self, config: CodexConfig, high_level: bool) -> list[str]:
        if high_level:
            async with AsyncCodex(config) as codex:
                assert codex.metadata.serverInfo is not None
                assert (await codex.account()).account is None
                await codex.logout()
                first = await codex.thread_start(sandbox=Sandbox.read_only)
                second = await codex.thread_start(sandbox=Sandbox.read_only)
                results = await asyncio.gather(
                    first.run("Say hello."), second.run("Say hello again.")
                )
                assert all(isinstance(result, TurnResult) for result in results)
                await first.set_name("Local test")
                assert (await first.read()).thread.id == first.id
                resumed = await codex.thread_resume(first.id, sandbox=Sandbox.read_only)
                forked = await codex.thread_fork(first.id, sandbox=Sandbox.read_only)
                assert resumed.id == first.id and forked.id != first.id
                assert (await codex.thread_list()).data
                await codex.thread_archive(forked.id)
                assert (await codex.thread_unarchive(forked.id)).id == forked.id
                return [result.final_response or "" for result in results]
        async with TemporalCodexClient(config) as client:
            metadata = await client.initialize()
            assert metadata.userAgent
            assert (await client.account_read()).account is None
            await client.account_logout()
            thread = await client.thread_start({"sandbox": "read-only"})
            assert isinstance(thread, ThreadStartResponse)
            pieces = [
                delta.delta
                async for delta in client.stream_text(thread.thread.id, "Say hello.")
            ]
            await client.thread_set_name(thread.thread.id, "Low-level test")
            assert (await client.thread_read(thread.thread.id, True)).thread.turns
            return ["".join(pieces)]


@workflow.defn
class FailedTurnWorkflow:
    @workflow.run
    async def run(self, config: CodexConfig) -> None:
        async with AsyncCodex(config) as codex:
            thread = await codex.thread_start(sandbox=Sandbox.read_only)
            await thread.run("Say hello.")


@workflow.defn
class StdoutFailureWorkflow:
    @workflow.run
    async def run(self, config: CodexConfig) -> None:
        async with TemporalCodexClient(config) as client:
            await client.next_notification()


@workflow.defn
class PendingWorkflow:
    @workflow.run
    async def run(self) -> None:
        async with TemporalCodexClient():
            await workflow.wait_condition(lambda: False)


@workflow.defn
class ConnectingWorkflow:
    @workflow.run
    async def run(self, queue: str | None, abandon: bool) -> None:
        connect: workflow.ActivityConfig = {
            "retry_policy": RetryPolicy(initial_interval=timedelta(minutes=1)),
            "heartbeat_timeout": timedelta(milliseconds=100),
        }
        if queue is not None:
            connect["task_queue"] = queue
        if abandon:
            connect["cancellation_type"] = workflow.ActivityCancellationType.ABANDON
        async with TemporalCodexClient(activity_options={"connect": connect}):
            await workflow.wait_condition(lambda: False)


@workflow.defn
class ClosingApprovalWorkflow:
    @workflow.run
    async def run(self) -> bool:
        finished = False
        client: TemporalCodexClient

        async def approve(method: str, params: JsonObject | None) -> JsonObject:
            nonlocal finished
            del method, params
            await client.close()
            finished = True
            return {"decision": "accept"}

        async with TemporalCodexClient(approval_handler=approve) as client:
            try:
                await client.request("approval", None, response_model=Echo)
            except TransportClosedError:
                await workflow.wait_condition(lambda: finished)
                return finished
        raise AssertionError("approval callback should have closed the client")


@workflow.defn
class ProtocolWorkflow:
    @workflow.run
    async def run(self, asynchronous: bool) -> list[str]:
        client: TemporalCodexClient
        calls: list[str] = []

        def sync_approve(method: str, params: JsonObject | None) -> JsonObject:
            assert params == {"command": "echo test"}
            calls.append(method)
            return {"decision": "accept"}

        async def async_approve(method: str, params: JsonObject | None) -> JsonObject:
            assert (await client.next_notification()).method == "probe"
            # Reentrant RPC proves approvals do not block the message reader.
            assert (
                await client.request("echo", {"value": "nested"}, response_model=Echo)
            ).value == "nested"
            return sync_approve(method, params)

        async with TemporalCodexClient(
            approval_handler=async_approve if asynchronous else sync_approve
        ) as client:
            response = await client.request("approval", None, response_model=Echo)
            assert response.value == "accept"
            try:
                await client.request("fail", None, response_model=Echo)
            except InvalidParamsError as error:
                assert error.code == -32602 and error.data == {"field": "value"}
            else:
                raise AssertionError("expected typed RPC failure")
            return calls


@workflow.defn
class ConnectionLossWorkflow:
    @workflow.run
    async def run(self) -> str:
        async with TemporalCodexClient(
            activity_options={
                "read": {"schedule_to_start_timeout": timedelta(seconds=1)},
                "write": {"schedule_to_start_timeout": timedelta(seconds=1)},
            }
        ) as client:
            await workflow.wait_condition(lambda: self.proceed)
            try:
                await client.request("echo", {"value": "late"}, response_model=Echo)
            except Exception as error:
                return str(error)
        return "unexpected success"

    def __init__(self) -> None:
        self.proceed = False

    @workflow.signal
    def resume(self) -> None:
        self.proceed = True


@workflow.defn
class ControlWorkflow:
    @workflow.run
    async def run(self, config: CodexConfig, high_level: bool) -> str:
        if high_level:
            async with AsyncCodex(config) as codex:
                thread = await codex.thread_start(sandbox=Sandbox.read_only)
                turn = await thread.turn("Wait for me.")
                await workflow.wait_condition(lambda: self.proceed)
                await turn.steer("Additional context.")
                await turn.interrupt()
                return (await turn.run()).status.value
        async with TemporalCodexClient(config) as codex:
            await codex.initialize()
            started_thread = await codex.thread_start({"sandbox": "read-only"})
            started_turn = await codex.turn_start(
                started_thread.thread.id, "Wait for me."
            )
            await workflow.wait_condition(lambda: self.proceed)
            await codex.turn_steer(
                started_thread.thread.id, started_turn.turn.id, "Additional context."
            )
            await codex.turn_interrupt(started_thread.thread.id, started_turn.turn.id)
            return (
                await codex.wait_for_turn_completed(started_turn.turn.id)
            ).turn.status.value

    def __init__(self) -> None:
        self.proceed = False

    @workflow.signal
    def resume(self) -> None:
        self.proceed = True


@workflow.defn
class GoalWorkflow:
    @workflow.run
    async def run(self, config: CodexConfig) -> str:
        async with TemporalCodexClient(config) as codex:
            await codex.initialize()
            thread = await codex.thread_start({"sandbox": "read-only"})
            initial = await codex.turn_start(
                thread.thread.id, "Prepare the persisted thread."
            )
            # Discard the initial turn after waiting for its durable rollout.
            # The completion is emitted before the model has any goal set.
            await codex.wait_for_turn_completed(initial.turn.id)
            state, turn_id = await codex.start_goal_operation(
                thread.thread.id, "Say hello once."
            )
            assert turn_id
            await codex.cancel_goal_operation(state)
            assert (
                await codex.pause_goal(thread.thread.id)
            ).goal.status.value == "paused"
            await codex.thread_goal_clear(thread.thread.id)
            codex.unregister_goal_operation(state)
            return "paused"


@workflow.defn
class LoginWorkflow:
    @workflow.run
    async def run(self, high_level: bool, device: bool) -> bool:
        if high_level:
            async with AsyncCodex() as codex:
                login = (
                    await codex.login_chatgpt_device_code()
                    if device
                    else await codex.login_chatgpt()
                )
                assert login.login_id == "login"
                return (await login.wait()).success
        async with TemporalCodexClient() as client:
            await client.initialize()
            login_response = await client.account_login_start(
                {"type": "chatgptDeviceCode" if device else "chatgpt"}
            )
            login_id = getattr(login_response.root, "login_id")
            return (await client.wait_for_login_completed(login_id)).success


@workflow.defn
class DisconnectWorkflow:
    @workflow.run
    async def run(self) -> str:
        async with TemporalCodexClient() as client:
            try:
                await client.request("disconnect", None, response_model=Echo)
            except Exception as error:
                return str(error)
        return "unexpected success"


@workflow.defn
class InitializingWorkflow:
    @workflow.run
    async def run(self, lazy: bool) -> None:
        codex = AsyncCodex()
        if lazy:
            await codex.account()
        else:
            async with codex:
                pass


@workflow.defn
class JoiningTurnWorkflow:
    @workflow.run
    async def run(self, close_first: bool) -> list[str]:
        async with AsyncCodex() as codex:
            thread = AsyncThread(codex, "thread")
            first = await thread.turn("first input")
            joining = asyncio.create_task(thread.turn("additional input"))
            stream = first.stream()
            assert isinstance(stream, AsyncGenerator)
            try:
                assert (await stream.__anext__()).method == "item/agentMessage/delta"
                if close_first:
                    await stream.aclose()
                assert isinstance(codex._client, TemporalCodexClient)
                await codex._client.notify("release_join")
                joined = await joining
                if close_first:
                    try:
                        async for _ in first.stream():
                            raise AssertionError(
                                "A closed handle must not yield events"
                            )
                    except TransportClosedError:
                        pass
                events = [event.method async for event in joined.stream()]
                if not close_first:
                    assert [event.method async for event in stream] == [
                        "turn/completed"
                    ]
                return events
            finally:
                await stream.aclose()
