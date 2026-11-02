"""A controllable app-server connection for retry and callback tests."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from openai_codex import TransportClosedError
from pydantic import BaseModel

from temporalio.openai_codex import CodexProcess, ConnectActivityInput
from temporalio.openai_codex._types import Message


class Echo(BaseModel):
    value: str


class ProtocolProcess:
    def __init__(self) -> None:
        self.output: asyncio.Queue[list[Message]] = asyncio.Queue()
        self.writes: list[Message] = []
        self.closed = False
        self.approval_request: Any = None
        self.gate: asyncio.Event | None = None
        self.reads = 0

    async def write(self, message: Message) -> None:
        self.writes.append(message)
        if self.gate is not None:
            await self.gate.wait()
        if message.get("method") == "disconnect":
            raise TransportClosedError("broken pipe")
        if message.get("method") == "approval":
            self.approval_request = message["id"]
            self.output.put_nowait(
                [
                    {
                        "id": "approve",
                        "method": "item/commandExecution/requestApproval",
                        "params": {"command": "echo test"},
                    },
                    {"method": "probe", "params": {}},
                ]
            )
        elif message.get("id") == "approve":
            self.output.put_nowait(
                [
                    {
                        "id": self.approval_request,
                        "result": {"value": message["result"]["decision"]},
                    }
                ]
            )
        elif message.get("method") == "fail":
            self.output.put_nowait(
                [
                    {
                        "id": message["id"],
                        "error": {
                            "code": -32602,
                            "message": "bad input",
                            "data": {"field": "value"},
                        },
                    }
                ]
            )
        elif message.get("method") == "echo":
            self.output.put_nowait([{"id": message["id"], "result": message["params"]}])

        elif message.get("method") == "initialize":
            self.output.put_nowait(
                [{"id": message["id"], "result": {"userAgent": "codex/0.163.0"}}]
            )
        elif message.get("method") == "account/login/start":
            device = message["params"]["type"] == "chatgptDeviceCode"
            response = (
                {
                    "type": "chatgptDeviceCode",
                    "loginId": "login",
                    "verificationUrl": "https://example.invalid",
                    "userCode": "TEST",
                }
                if device
                else {
                    "type": "chatgpt",
                    "loginId": "login",
                    "authUrl": "https://example.invalid",
                }
            )
            self.output.put_nowait(
                [
                    {
                        "method": "account/login/completed",
                        "params": {"loginId": "login", "success": True, "error": None},
                    },
                    {"id": message["id"], "result": response},
                ]
            )

    async def read(self) -> list[Message]:
        self.reads += 1
        return await self.output.get()

    async def close(self) -> None:
        self.closed = True
        self.output.put_nowait([])


class ProtocolConnector:
    def __init__(
        self, process_factory: Callable[[], ProtocolProcess] = ProtocolProcess
    ) -> None:
        self.processes: list[ProtocolProcess] = []
        self.connects = 0
        self._process_factory = process_factory

    async def connect(self, request: ConnectActivityInput) -> CodexProcess:
        del request
        self.connects += 1
        process = self._process_factory()
        self.processes.append(process)
        return process
