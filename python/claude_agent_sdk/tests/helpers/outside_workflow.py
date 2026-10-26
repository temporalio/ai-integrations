"""Run the agent's Workflow helpers outside a Workflow (tests only).

A few helpers decide how big things are or whether to stop: they call
``workflow.payload_converter()``, ``workflow.info()``, ``workflow.patched()``,
``workflow.logger`` and, to hand over, ``workflow.continue_as_new()``, which only
work in a Workflow. ``OutsideWorkflow`` puts the default converter, a history of your
choice, a patch answer, a plain logger and a recorded hand-over in their place, so a
test can check such a helper over thousands of cases in seconds. Each helper tested
this way also runs inside a real Workflow in other tests.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest

from temporalio import workflow
from temporalio.contrib.workflow_streams import WorkflowStreamState
from temporalio.converter import DataConverter


@dataclass
class History:
    """The current run's history as ``workflow.info()`` reports it."""

    events: int = 0
    size: int = 0
    suggested: bool = False  # the server suggests Continue-As-New

    def get_current_history_length(self) -> int:
        return self.events

    def get_current_history_size(self) -> int:
        return self.size

    def is_continue_as_new_suggested(self) -> bool:
        return self.suggested


class ContinuedAsNew(Exception):
    """Raised where ``workflow.continue_as_new`` would end the run."""

    def __init__(self, run_args: list[Any]) -> None:
        super().__init__("continued as new")
        self.run_args = run_args


@dataclass
class OutsideWorkflow:
    """What the replaced ``workflow`` functions answer; change it between calls."""

    history: History = field(default_factory=History)
    patched: bool = True  # whether this run makes the checks behind a patch
    answers: dict[str, bool] = field(default_factory=dict)  # per patch ID, if not that
    patches: list[str] = field(default_factory=list)  # the patch IDs asked for

    def install(self, monkeypatch: pytest.MonkeyPatch) -> OutsideWorkflow:
        """Replace the ``workflow`` functions for the rest of the test."""
        converter = DataConverter.default.payload_converter
        monkeypatch.setattr(workflow, "payload_converter", lambda: converter)
        monkeypatch.setattr(workflow, "info", lambda: self.history)
        monkeypatch.setattr(workflow, "logger", logging.getLogger("outside-workflow"))

        def patched(patch_id: str) -> bool:
            self.patches.append(patch_id)
            return self.answers.get(patch_id, self.patched)

        monkeypatch.setattr(workflow, "patched", patched)

        async def wait_condition(ready: Callable[[], bool], **_: Any) -> None:
            assert ready(), "a Workflow would wait here forever"

        def continue_as_new(**options: Any) -> None:
            raise ContinuedAsNew(list(options["args"]))

        monkeypatch.setattr(workflow, "wait_condition", wait_condition)
        monkeypatch.setattr(workflow, "all_handlers_finished", lambda: True)
        monkeypatch.setattr(workflow, "continue_as_new", continue_as_new)
        now = datetime(2026, 10, 3, tzinfo=timezone.utc)
        monkeypatch.setattr(workflow, "now", lambda: now)
        return self


class CarriedStream:
    """What ``_carry_stream`` uses of a Workflow's stream, outside a Workflow."""

    def __init__(self, state: WorkflowStreamState) -> None:
        self.state = state
        self.detached = False

    def get_state(self) -> WorkflowStreamState:
        return dataclasses.replace(self.state, log=list(self.state.log))

    def detach_pollers(self) -> None:
        self.detached = True

    def truncate(self, up_to_offset: int) -> None:
        dropped = up_to_offset - self.state.base_offset
        self.state = dataclasses.replace(
            self.state, log=self.state.log[dropped:], base_offset=up_to_offset
        )


class Topic:
    """What the agent publishes to, outside a Workflow: the events, in order."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def publish(self, event: dict[str, Any]) -> None:
        self.events.append(event)
