"""Deterministic Codex JSON-RPC client for Temporal Workflows."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import timedelta
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from openai_codex import CodexConfig, CodexError, TransportClosedError
from openai_codex._initialize_metadata import _split_user_agent
from openai_codex._runtime_requirements import require_runtime_version
from openai_codex.client import _active_turn_id_from_error, _params_dict
from openai_codex.errors import (
    InvalidRequestError,
    is_retryable_error,
    map_jsonrpc_error,
)
from openai_codex.generated.notification_registry import NOTIFICATION_MODELS
from openai_codex.generated.v2_all import (
    AccountLoginCompletedNotification,
    AgentMessageDeltaNotification,
    CancelLoginAccountResponse,
    GetAccountResponse,
    IdleThreadStatus,
    LoginAccountResponse,
    LogoutAccountResponse,
    ModelListResponse,
    ThreadArchiveResponse,
    ThreadCompactStartResponse,
    ThreadForkResponse,
    ThreadGoalClearResponse,
    ThreadGoalSetResponse,
    ThreadGoalStatus,
    ThreadListResponse,
    ThreadReadResponse,
    ThreadResumeResponse,
    ThreadSetNameResponse,
    ThreadStartResponse,
    ThreadUnarchiveResponse,
    TurnCompletedNotification,
    TurnInterruptResponse,
    TurnStartResponse,
    TurnSteerResponse,
)
from openai_codex.generated.v2_all import GetAccountParams as V2GetAccountParams
from openai_codex.generated.v2_all import LoginAccountParams as V2LoginAccountParams
from openai_codex.generated.v2_all import ThreadForkParams as V2ThreadForkParams
from openai_codex.generated.v2_all import ThreadListParams as V2ThreadListParams
from openai_codex.generated.v2_all import ThreadResumeParams as V2ThreadResumeParams
from openai_codex.generated.v2_all import ThreadStartParams as V2ThreadStartParams
from openai_codex.generated.v2_all import TurnStartParams as V2TurnStartParams
from openai_codex.models import (
    InitializeResponse,
    JsonObject,
    Notification,
    UnknownNotification,
)
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, is_cancelled_exception

from ._constants import CLOSE, CONNECT, CONNECTION_LOST, READ, RESERVE, WRITE
from ._errors import sdk_error
from ._routing import GoalOperation, Router, Subscription
from ._types import (
    CloseActivityInput,
    CodexActivityOptions,
    ConnectActivityInput,
    Connection,
    Message,
    ReadActivityInput,
    ReadResult,
    WriteActivityInput,
)

ModelT = TypeVar("ModelT", bound=BaseModel)
ApprovalHandler = Callable[[str, JsonObject | None], JsonObject | Awaitable[JsonObject]]


class TemporalCodexClient:
    """The AsyncCodexClient operations implemented with Workflow-safe primitives.

    Entering the context starts the connection. Call initialize explicitly before
    other protocol operations, just as with upstream AsyncCodexClient.
    """

    def __init__(
        self,
        config: CodexConfig | None = None,
        approval_handler: ApprovalHandler | None = None,
        *,
        activity_options: CodexActivityOptions | None = None,
    ) -> None:
        """Configure a connection; no process starts until start or context entry."""
        self.config = config or CodexConfig()
        self._approval_handler = approval_handler
        self._activity_options: CodexActivityOptions = activity_options or {}
        self._connection: Connection | None = None
        self._lost = False
        self._reader: asyncio.Task[None] | None = None
        self._callbacks: dict[asyncio.Task[None], None] = {}
        self._closing: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[Any]] = {}
        self._router = Router()
        self._request_id = 0
        self._write_id = 0
        self._received = 0
        self._runtime_version: str | None = None
        self._write_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._thread_locks: dict[str, asyncio.Lock] = {}

    async def __aenter__(self) -> TemporalCodexClient:
        """Start the connection without implicitly initializing the protocol."""
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Close the connection on normal completion, error, or cancellation."""
        if not args or args[0] is not GeneratorExit:
            await self.close()

    async def start(self) -> None:
        """Launch once, routing all subsequent Activities to the owning Worker."""
        if not workflow.in_workflow():
            raise RuntimeError("TemporalCodexClient must run in a Temporal Workflow")
        async with self._start_lock:
            if self._connection is not None:
                return
            if self._lost:
                raise TransportClosedError(CONNECTION_LOST)
            reservation_options = self._activity_options.get("connect", {}).copy()
            reservation_options.pop("activity_id", None)
            reserving = asyncio.create_task(
                self._execute(
                    RESERVE,
                    None,
                    reservation_options,
                    Connection,
                )
            )
            try:
                connection: Connection = await asyncio.shield(reserving)
            except asyncio.CancelledError:
                reserving.cancel()
                await asyncio.gather(reserving, return_exceptions=True)
                raise
            self._connection = connection
            connecting = asyncio.create_task(
                self._execute(
                    CONNECT,
                    ConnectActivityInput(self.config, connection.handle),
                    self._activity_options.get("connect", {}),
                    Connection,
                )
            )
            try:
                await asyncio.shield(connecting)
            except (Exception, asyncio.CancelledError):
                # Stop queued/retrying launches. The owner and identity are
                # already known, so close also covers late committed results.
                connecting.cancel()
                await asyncio.gather(connecting, return_exceptions=True)
                await self.close()
                raise
            self._router = Router()
            self._received = 0
            self._reader = asyncio.create_task(self._read_loop())

    async def close(self) -> None:
        """Close Activities and Workflow tasks, also during Workflow cancellation."""
        if self._connection is None:
            return
        # A separate task can finish cleanup while the calling task is cancelled.
        caller = asyncio.current_task()
        if self._closing is None:
            self._closing = asyncio.create_task(self._close(caller))
        elif caller in self._callbacks:
            # Another cleanup may already be joining this callback.
            return
        cleanup = self._closing
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise
        finally:
            if cleanup.done() and self._closing is cleanup:
                self._closing = None

    async def _close(self, caller: asyncio.Task[Any] | None) -> None:
        reader, self._reader = self._reader, None
        tasks = [task for task in self._callbacks if task is not caller]
        if reader is not None:
            tasks.append(reader)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            if not self._lost and self._connection is not None:
                await self._execute(
                    CLOSE,
                    CloseActivityInput(self._connection.handle),
                    self._activity_options.get("close", {}),
                )
        finally:
            self._connection = None
            self._runtime_version = None
            self._fail(TransportClosedError("Codex process is not running"))

    async def _execute(
        self,
        name: str,
        arg: Any,
        config: workflow.ActivityConfig,
        result_type: Any = None,
    ) -> Any:
        if self._lost:
            raise TransportClosedError(CONNECTION_LOST)
        options: workflow.ActivityConfig = {
            "start_to_close_timeout": timedelta(minutes=5),
            "heartbeat_timeout": timedelta(seconds=10),
            "retry_policy": RetryPolicy(backoff_coefficient=1.0),
            **config,
        }
        if name != RESERVE:
            assert self._connection is not None
            options.setdefault("schedule_to_start_timeout", timedelta(seconds=30))
            options["task_queue"] = self._connection.task_queue
        try:
            return await workflow.execute_activity(
                name, arg, result_type=result_type, **options
            )
        except ActivityError as error:
            if is_cancelled_exception(error):
                raise asyncio.CancelledError() from error
            sdk = sdk_error(error)
            if isinstance(sdk, TransportClosedError) and str(sdk) == CONNECTION_LOST:
                self._lost = True
            raise sdk from error

    async def _write(self, message: Message) -> None:
        async with self._write_lock:
            if self._connection is None:
                raise TransportClosedError("Codex process is not running")
            self._write_id += 1
            await self._execute(
                WRITE,
                WriteActivityInput(self._connection.handle, self._write_id, message),
                self._activity_options.get("write", {}),
            )

    async def _read_loop(self) -> None:
        try:
            while self._connection is not None:
                result: ReadResult = await self._execute(
                    READ,
                    ReadActivityInput(self._connection.handle, self._received),
                    self._activity_options.get("read", {}),
                    ReadResult,
                )
                if not result.output:
                    raise TransportClosedError("Codex process closed stdout")
                self._received += len(result.output)
                for message in result.output:
                    if "method" in message:
                        if "id" in message:
                            task = asyncio.create_task(self._approve(message))
                            self._callbacks[task] = None
                            task.add_done_callback(
                                lambda completed: self._callbacks.pop(completed, None)
                            )
                        else:
                            method = message["method"]
                            params = message.get("params", {})
                            model = NOTIFICATION_MODELS.get(method)
                            try:
                                payload = (
                                    model.model_validate(params)
                                    if model
                                    else UnknownNotification(params=params)
                                )
                            except Exception:
                                payload = UnknownNotification(
                                    params=params if isinstance(params, dict) else {}
                                )
                            self._router.route(
                                Notification(method=method, payload=payload)
                            )
                    elif waiter := self._pending.get(str(message.get("id"))):
                        if waiter.done():
                            continue
                        if "error" in message:
                            error = message["error"]
                            waiter.set_exception(
                                map_jsonrpc_error(
                                    error.get("code", -32000),
                                    error.get("message", "unknown"),
                                    error.get("data"),
                                )
                            )
                        else:
                            waiter.set_result(message.get("result"))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._fail(error)

    def _fail(self, error: BaseException) -> None:
        for waiter in self._pending.values():
            if not waiter.done():
                waiter.set_exception(error)
        self._router.fail(error)

    async def _approve(self, message: Message) -> None:
        try:
            method = message["method"]
            params = message.get("params")
            if self._approval_handler is None:
                result: Any = (
                    {"decision": "accept"}
                    if method
                    in {
                        "item/commandExecution/requestApproval",
                        "item/fileChange/requestApproval",
                    }
                    else {}
                )
            else:
                result = self._approval_handler(
                    method, params if isinstance(params, dict) else None
                )
                if inspect.isawaitable(result):
                    result = await result
            await self._write({"id": message["id"], "result": result})
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._fail(error)

    async def request(
        self, method: str, params: JsonObject | None, *, response_model: type[ModelT]
    ) -> ModelT:
        """Send a typed request with deterministic IDs and upstream RPC errors."""
        if self._router.error is not None:
            raise self._router.error
        runtime_fields = {
            "turn/start": ("toolOutput", "turnTrigger", "serviceTierForTurn"),
            "thread/resume": ("excludeTurns",),
            "thread/fork": ("excludeTurns",),
        }
        if any(
            (params or {}).get(key) is not None
            for key in runtime_fields.get(method, ())
        ):
            try:
                require_runtime_version(self._runtime_version)
            except ValueError as error:
                raise CodexError(str(error)) from error
        self._request_id += 1
        request_id = str(self._request_id)
        waiter = asyncio.get_running_loop().create_future()
        self._pending[request_id] = waiter
        message: Message = {"id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        try:
            await self._write(message)
            result = await waiter
            if not isinstance(result, dict):
                raise CodexError(f"{method} response must be a JSON object")
            return response_model.model_validate(result)
        finally:
            self._pending.pop(request_id, None)
            if waiter.done() and not waiter.cancelled():
                waiter.exception()  # Observe failures even if writing failed first.

    async def notify(self, method: str, params: JsonObject | None = None) -> None:
        """Send a notification without allocating a response waiter."""
        message: Message = {"method": method}
        if params is not None:
            message["params"] = params
        await self._write(message)

    async def initialize(self) -> InitializeResponse:
        """Perform the explicit app-server initialization handshake."""
        result = await self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": self.config.client_name,
                    "title": self.config.client_title,
                    "version": self.config.client_version,
                },
                "capabilities": {"experimentalApi": self.config.experimental_api},
            },
            response_model=InitializeResponse,
        )
        version = result.serverInfo.version if result.serverInfo else None
        if not version or not version.strip():
            _, version = _split_user_agent(result.userAgent or "")
        self._runtime_version = (
            version.split()[0] if version and version.strip() else None
        )
        await self.notify("initialized")
        return result

    def _subscribe_turn_notifications(self, turn_id: str) -> Subscription:
        return self._router.subscribe(turn_id)

    def register_turn_notifications(self, turn_id: str) -> None:
        """Register the low-level turn consumer."""
        self._router.register_turn(turn_id)

    def unregister_turn_notifications(self, turn_id: str) -> None:
        """Release the low-level turn consumer."""
        self._router.unregister_turn(turn_id)

    def register_login_notifications(self, login_id: str) -> None:
        """Register a login's completion route."""
        self._router.login(login_id)

    def unregister_login_notifications(self, login_id: str) -> None:
        """Release a login's notification route."""
        self._router.logins.pop(login_id, None)

    async def next_notification(self) -> Notification:
        """Wait for a notification outside registered turn and login routes."""
        return await self._router.global_inbox.next()

    async def next_turn_notification(self, turn_id: str) -> Notification:
        """Wait for the low-level consumer's next turn event."""
        if turn_id not in self._router.defaults:
            raise RuntimeError(f"turn {turn_id!r} is not registered for streaming")
        return await self._router.defaults[turn_id].next()

    async def next_login_notification(self, login_id: str) -> Notification:
        """Wait for one login's next event."""
        if login_id not in self._router.logins:
            raise RuntimeError(f"login {login_id!r} is not registered for waiting")
        return await self._router.logins[login_id].next()

    async def _start_turn(
        self,
        thread_id: str,
        input_items: list[JsonObject] | JsonObject | str,
        params: Any,
        for_handle: bool,
    ) -> tuple[Any, Subscription | None]:
        async with self._thread_locks.setdefault(thread_id, asyncio.Lock()):
            if thread_id in self._router.goals:
                raise InvalidRequestError(
                    -32600, f"thread has an active goal operation: {thread_id}"
                )
            cursors = {key: state.end for key, state in self._router.turns.items()}
            self._router.pending += 1
            try:
                result = await self.request(
                    "turn/start",
                    {
                        **_params_dict(params),
                        "threadId": thread_id,
                        "input": cast(Any, self._normalize_input_items(input_items)),
                    },
                    response_model=TurnStartResponse,
                )
                turn_id = result.turn.id
                if for_handle:
                    subscription = self._router.subscribe(
                        turn_id, cursor=cursors.get(turn_id, 0)
                    )
                else:
                    if turn_id not in self._router.defaults:
                        self._router.defaults[turn_id] = self._router.subscribe(
                            turn_id, cursor=cursors.get(turn_id, 0)
                        )
                    subscription = None
                return result, subscription
            finally:
                self._router.pending -= 1
                self._router.prune()

    async def turn_start(
        self,
        thread_id: str,
        input_items: list[JsonObject] | JsonObject | str,
        params: V2TurnStartParams | JsonObject | None = None,
    ) -> TurnStartResponse:
        """Start a turn and subscribe before early output can be discarded."""
        return (await self._start_turn(thread_id, input_items, params, False))[0]

    @staticmethod
    def _normalize_input_items(
        input_items: list[JsonObject] | JsonObject | str,
    ) -> list[JsonObject]:
        if isinstance(input_items, str):
            return [{"type": "text", "text": input_items}]
        return [input_items] if isinstance(input_items, dict) else input_items

    async def request_with_retry_on_overload(
        self,
        method: str,
        params: JsonObject | None,
        *,
        response_model: type[ModelT],
        max_attempts: int = 3,
        initial_delay_s: float = 0.25,
        max_delay_s: float = 2.0,
    ) -> ModelT:
        """Retry server overload with deterministic jitter and Workflow timers."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        delay = initial_delay_s
        for attempt in range(max_attempts):
            try:
                return await self.request(method, params, response_model=response_model)
            except Exception as error:
                if attempt + 1 == max_attempts or not is_retryable_error(error):
                    raise
                await asyncio.sleep(
                    max(
                        0,
                        min(max_delay_s, delay)
                        + workflow.random().uniform(-delay * 0.2, delay * 0.2),
                    )
                )
                delay = min(max_delay_s, delay * 2)
        raise AssertionError("unreachable")

    def register_goal_operation(self, thread_id: str) -> GoalOperation:
        """Register a thread-scoped goal route."""
        state = self._reserve_goal(thread_id)
        state.routing = True
        return state

    def _reserve_goal(self, thread_id: str) -> GoalOperation:
        if thread_id in self._router.goals:
            raise RuntimeError(
                f"thread {thread_id!r} already has an active goal operation"
            )
        state = GoalOperation(thread_id)
        if self._router.error is not None:
            state.inbox.fail(self._router.error)
        self._router.goals[thread_id] = state
        return state

    def unregister_goal_operation(self, state: GoalOperation) -> None:
        """Release one goal operation's route."""
        if self._router.goals.get(state.thread_id) is state:
            self._router.goals.pop(state.thread_id)

    async def next_goal_notification(self, state: GoalOperation) -> Notification:
        """Wait for a notification routed to a goal operation."""
        return await state.inbox.next()

    async def start_goal_operation(
        self, thread_id: str, objective: str
    ) -> tuple[GoalOperation, str]:
        """Set a persisted goal and wait for its runtime-generated first turn."""
        async with self._thread_locks.setdefault(thread_id, asyncio.Lock()):
            thread = (await self.thread_read(thread_id)).thread
            if not isinstance(thread.status.root, IdleThreadStatus):
                raise InvalidRequestError(
                    -32600, f"thread must be idle before starting a goal: {thread_id}"
                )
            if thread.ephemeral or thread.path is None:
                raise InvalidRequestError(
                    -32600,
                    f"thread must be persisted before starting a goal: {thread_id}",
                )
            state = self._reserve_goal(thread_id)
            try:
                await self.thread_goal_clear(thread_id)
                state.routing = True
                await self.thread_goal_set(
                    thread_id, objective=objective, status=ThreadGoalStatus.active
                )
                try:
                    await workflow.wait_condition(
                        lambda: (
                            state.logical_turn_id is not None
                            or state.inbox.error is not None
                        ),
                        timeout=30,
                    )
                except asyncio.TimeoutError as error:
                    raise CodexError(
                        "timed out waiting for goal turn to start after 30 seconds"
                    ) from error
                if state.inbox.error is not None:
                    raise state.inbox.error
                assert state.logical_turn_id is not None
                return state, state.logical_turn_id
            except BaseException:
                await self.cancel_goal_operation(state)
                state.finish()
                self.unregister_goal_operation(state)
                raise

    async def cancel_goal_operation(self, state: GoalOperation) -> None:
        """Pause continuation and interrupt the active physical turn when possible."""
        try:
            await self.pause_goal(state.thread_id)
        except Exception:
            pass
        turn_id = (
            state.current_turn_id
            or getattr(state.completed_turn, "id", None)
            or state.logical_turn_id
        )
        if turn_id is None:
            return
        try:
            await self.turn_interrupt(state.thread_id, turn_id)
        except InvalidRequestError as error:
            active = _active_turn_id_from_error(error)
            if active and active != turn_id:
                try:
                    await self.turn_interrupt(state.thread_id, active)
                except Exception:
                    pass
        except Exception:
            pass

    async def account_login_start(
        self, params: V2LoginAccountParams | JsonObject
    ) -> LoginAccountResponse:
        """Send the typed account/login/start request."""
        result = await self.request(
            "account/login/start",
            _params_dict(params),
            response_model=LoginAccountResponse,
        )
        login_id = getattr(result.root, "login_id", None)
        if login_id is not None:
            self.register_login_notifications(login_id)
        return result

    async def account_login_cancel(self, login_id: str) -> CancelLoginAccountResponse:
        """Send the typed account/login/cancel request."""
        result = await self.request(
            "account/login/cancel",
            {"loginId": login_id},
            response_model=CancelLoginAccountResponse,
        )
        return result

    async def account_read(
        self, params: V2GetAccountParams | JsonObject | None = None
    ) -> GetAccountResponse:
        """Send the typed account/read request."""
        result = await self.request(
            "account/read", _params_dict(params), response_model=GetAccountResponse
        )
        return result

    async def account_logout(self) -> LogoutAccountResponse:
        """Send the typed account/logout request."""
        result = await self.request(
            "account/logout", None, response_model=LogoutAccountResponse
        )
        return result

    async def thread_start(
        self, params: V2ThreadStartParams | JsonObject | None = None
    ) -> ThreadStartResponse:
        """Send the typed thread/start request."""
        result = await self.request(
            "thread/start", _params_dict(params), response_model=ThreadStartResponse
        )
        return result

    async def thread_resume(
        self, thread_id: str, params: V2ThreadResumeParams | JsonObject | None = None
    ) -> ThreadResumeResponse:
        """Send the typed thread/resume request."""
        result = await self.request(
            "thread/resume",
            {"threadId": thread_id, **_params_dict(params)},
            response_model=ThreadResumeResponse,
        )
        return result

    async def thread_list(
        self, params: V2ThreadListParams | JsonObject | None = None
    ) -> ThreadListResponse:
        """Send the typed thread/list request."""
        result = await self.request(
            "thread/list", _params_dict(params), response_model=ThreadListResponse
        )
        return result

    async def thread_read(
        self, thread_id: str, include_turns: bool = False
    ) -> ThreadReadResponse:
        """Send the typed thread/read request."""
        result = await self.request(
            "thread/read",
            {"threadId": thread_id, "includeTurns": include_turns},
            response_model=ThreadReadResponse,
        )
        return result

    async def thread_fork(
        self, thread_id: str, params: V2ThreadForkParams | JsonObject | None = None
    ) -> ThreadForkResponse:
        """Send the typed thread/fork request."""
        result = await self.request(
            "thread/fork",
            {"threadId": thread_id, **_params_dict(params)},
            response_model=ThreadForkResponse,
        )
        return result

    async def thread_archive(self, thread_id: str) -> ThreadArchiveResponse:
        """Send the typed thread/archive request."""
        result = await self.request(
            "thread/archive",
            {"threadId": thread_id},
            response_model=ThreadArchiveResponse,
        )
        return result

    async def thread_unarchive(self, thread_id: str) -> ThreadUnarchiveResponse:
        """Send the typed thread/unarchive request."""
        result = await self.request(
            "thread/unarchive",
            {"threadId": thread_id},
            response_model=ThreadUnarchiveResponse,
        )
        return result

    async def thread_set_name(self, thread_id: str, name: str) -> ThreadSetNameResponse:
        """Send the typed thread/name/set request."""
        result = await self.request(
            "thread/name/set",
            {"threadId": thread_id, "name": name},
            response_model=ThreadSetNameResponse,
        )
        return result

    async def thread_compact(self, thread_id: str) -> ThreadCompactStartResponse:
        """Send the typed thread/compact/start request."""
        result = await self.request(
            "thread/compact/start",
            {"threadId": thread_id},
            response_model=ThreadCompactStartResponse,
        )
        return result

    async def thread_goal_clear(self, thread_id: str) -> ThreadGoalClearResponse:
        """Send the typed thread/goal/clear request."""
        result = await self.request(
            "thread/goal/clear",
            {"threadId": thread_id},
            response_model=ThreadGoalClearResponse,
        )
        return result

    async def turn_interrupt(
        self, thread_id: str, turn_id: str
    ) -> TurnInterruptResponse:
        """Send the typed turn/interrupt request."""
        result = await self.request(
            "turn/interrupt",
            {"threadId": thread_id, "turnId": turn_id},
            response_model=TurnInterruptResponse,
        )
        return result

    async def turn_steer(
        self,
        thread_id: str,
        expected_turn_id: str,
        input_items: list[JsonObject] | JsonObject | str,
    ) -> TurnSteerResponse:
        """Send the typed turn/steer request."""
        result = await self.request(
            "turn/steer",
            {
                "threadId": thread_id,
                "expectedTurnId": expected_turn_id,
                "input": cast(Any, self._normalize_input_items(input_items)),
            },
            response_model=TurnSteerResponse,
        )
        return result

    async def model_list(self, include_hidden: bool = False) -> ModelListResponse:
        """Send the typed model/list request."""
        result = await self.request(
            "model/list",
            {"includeHidden": include_hidden},
            response_model=ModelListResponse,
        )
        return result

    async def thread_goal_set(
        self,
        thread_id: str,
        *,
        objective: str | None = None,
        status: ThreadGoalStatus | None = None,
    ) -> ThreadGoalSetResponse:
        """Create or update a thread's persisted goal."""
        params: JsonObject = {"threadId": thread_id}
        if objective is not None:
            params["objective"] = objective
        if status is not None:
            params["status"] = status.value
        return await self.request(
            "thread/goal/set", params, response_model=ThreadGoalSetResponse
        )

    async def pause_goal(self, thread_id: str) -> ThreadGoalSetResponse:
        """Pause the active persisted goal."""
        return await self.thread_goal_set(thread_id, status=ThreadGoalStatus.paused)

    async def wait_for_turn_completed(self, turn_id: str) -> TurnCompletedNotification:
        """Consume the low-level turn route through its completion notification."""
        self.register_turn_notifications(turn_id)
        try:
            while True:
                event = await self.next_turn_notification(turn_id)
                if (
                    isinstance(event.payload, TurnCompletedNotification)
                    and event.payload.turn.id == turn_id
                ):
                    return event.payload
        finally:
            self.unregister_turn_notifications(turn_id)

    async def wait_for_login_completed(
        self, login_id: str
    ) -> AccountLoginCompletedNotification:
        """Consume a login route through its matching completion notification."""
        self.register_login_notifications(login_id)
        try:
            while True:
                event = await self.next_login_notification(login_id)
                if (
                    isinstance(event.payload, AccountLoginCompletedNotification)
                    and event.payload.login_id == login_id
                ):
                    return event.payload
        finally:
            self.unregister_login_notifications(login_id)

    async def stream_text(
        self,
        thread_id: str,
        text: str,
        params: V2TurnStartParams | JsonObject | None = None,
    ) -> AsyncIterator[AgentMessageDeltaNotification]:
        """Start a turn and yield only its typed agent-message deltas."""
        started = await self.turn_start(thread_id, text, params)
        try:
            while True:
                event = await self.next_turn_notification(started.turn.id)
                if isinstance(event.payload, AgentMessageDeltaNotification):
                    yield event.payload
                elif isinstance(event.payload, TurnCompletedNotification):
                    return
        finally:
            self.unregister_turn_notifications(started.turn.id)
