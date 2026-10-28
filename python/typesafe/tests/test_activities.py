"""The ``system_one`` Activity against a fake TypeSafe HTTP transport."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import httpx2
import pytest
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from temporalio import activity
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment
from temporalio.typesafe._activities import TypeSafeActivities
from temporalio.typesafe._types import SystemOneInput, register_response_model
from temporalio.typesafe._workflow import ACTIVITY_SYSTEM_ONE
from tests.helpers.activity_info import fake_info
from tests.helpers.fake_typesafe import (
    BillingResponse,
    fake_response,
    fake_typesafe_responder,
    mock_client,
)


def _activities() -> tuple[TypeSafeActivities, AsyncTypeSafeClient]:
    client = mock_client()
    support = TypeSafeActivities(client)
    return support, client


def _activity(support: TypeSafeActivities) -> Any:
    """Return the registered ``system_one`` function by its Temporal definition name."""
    definitions = [
        (activity._Definition.from_callable(fn), fn) for fn in support.activities
    ]
    [(definition, function)] = definitions
    assert definition is not None
    assert definition.name == "temporalio.typesafe.system_one"
    return function


@pytest.mark.asyncio
async def test_system_one_sends_all_questions_in_one_request_and_returns_typed_dicts() -> (
    None
):
    captured: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> Any:
        captured.append(body)
        return fake_response(
            200,
            {
                "model": "jev-1.13.0",
                "answers": {
                    "spam": {"type": "noul", "noul": 0.1},
                    "tone": {
                        "type": "choice",
                        "choice": "calm",
                        "confidence": 0.95,
                        "probabilities": {"calm": 0.95, "angry": 0.05},
                    },
                },
                "usage": {"input_tokens": 44, "output_tokens": 3},
            },
        )

    support = TypeSafeActivities(fake_typesafe_responder(handler))
    system_one = _activity(support)
    result = await system_one(
        SystemOneInput(
            model_name=None,
            state={"message": "Hello"},
            questions={
                "spam": {"type": "noul", "instructions": "Is this spam?"},
                "tone": {"type": "choice", "criteria": {"calm": None, "angry": None}},
            },
        )
    )
    assert result["model"] == "jev-1.13.0"
    assert result["answers"]["spam"] == {"type": "noul", "noul": 0.1}
    assert result["answers"]["tone"]["choice"] == "calm"
    assert result["usage"] == {"input_tokens": 44, "output_tokens": 3}
    [(sent)] = captured
    assert set(sent["questions"]) == {"spam", "tone"}


async def test_activity_name_matches_workflow_proxy_constant() -> None:
    support, _ = _activities()
    system_one = _activity(support)
    definition = activity._Definition.from_callable(system_one)
    assert definition is not None
    assert definition.name == ACTIVITY_SYSTEM_ONE


@pytest.mark.asyncio
async def test_http_500_is_retryable_through_activity() -> None:
    support = TypeSafeActivities(mock_client(status=500))
    system_one = _activity(support)
    with pytest.raises(ApplicationError) as err:
        await system_one(
            SystemOneInput(
                model_name=None,
                state="state",
                questions={"s": {"type": "noul", "instructions": "yes?"}},
            )
        )
    assert err.value.non_retryable is False


@pytest.mark.asyncio
async def test_http_422_is_non_retryable_through_activity() -> None:
    support = TypeSafeActivities(mock_client(status=422))
    system_one = _activity(support)
    with pytest.raises(ApplicationError) as err:
        await system_one(
            SystemOneInput(
                model_name=None,
                state="state",
                questions={"s": {"type": "noul", "instructions": "Is this spam?"}},
            )
        )
    assert err.value.non_retryable is True
    assert err.value.type == "TypeSafeRequestValidationError"


@pytest.mark.asyncio
async def test_state_passes_through_untouched() -> None:
    captured: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> Any:
        captured.append(body)
        return fake_response(
            200,
            {
                "model": "m",
                "answers": {"n": {"type": "noul", "noul": 0.5}},
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    support = TypeSafeActivities(fake_typesafe_responder(handler))
    system_one = _activity(support)
    state = {"nested": {"a": 1}, "items": [1, "two", None]}
    await system_one(
        SystemOneInput(
            model_name=None,
            state=state,
            questions={"n": {"type": "noul", "instructions": "Is this spam?"}},
        )
    )
    assert captured[0]["state"] == state


@pytest.mark.asyncio
async def test_unnamed_model_input_lets_client_default_answer() -> None:
    """A ``None`` model_name reaches the client, which fills its default.

    The recorded input stays exactly what the workflow named: nothing. The
    request body still carries the client's pin, so the served model shows
    up in the Activity result either way.
    """
    captured: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> Any:
        captured.append(body)
        return fake_response(
            200,
            {
                "model": "jev-pin",
                "answers": {"n": {"type": "noul", "noul": 0.5}},
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    support = TypeSafeActivities(fake_typesafe_responder(handler, model="jev-pin"))
    system_one = _activity(support)
    await system_one(
        SystemOneInput(
            model_name=None,
            state="state",
            questions={"n": {"type": "noul", "instructions": "Is this spam?"}},
        )
    )
    assert captured[0]["model"] == "jev-pin"


@pytest.mark.asyncio
async def test_named_model_input_is_sent_verbatim() -> None:
    """A named model overrides the client default on the request body."""
    captured: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> Any:
        captured.append(body)
        return fake_response(
            200,
            {
                "model": "jev-1.13.0",
                "answers": {"n": {"type": "noul", "noul": 0.5}},
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    support = TypeSafeActivities(fake_typesafe_responder(handler, model="jev-pin"))
    system_one = _activity(support)
    await system_one(
        SystemOneInput(
            state="state",
            questions={"n": {"type": "noul", "instructions": "Is this spam?"}},
            model_name="jev-1.13.0",
        )
    )
    assert captured[0]["model"] == "jev-1.13.0"


async def _system_one_request_id(client: AsyncTypeSafeClient) -> str | None:
    """Return the captured ``request_id`` for one call through a client."""
    system_one = _activity(TypeSafeActivities(client))
    result = await system_one(
        SystemOneInput(
            model_name=None,
            state="state",
            questions={"n": {"type": "noul", "instructions": "Is this spam?"}},
        )
    )
    return result["request_id"]


@pytest.mark.asyncio
async def test_registered_subclass_fields_survive_the_result_payload() -> None:
    """The Activity returns every field the registered subclass validates."""
    register_response_model("billing-activity", BillingResponse)
    support = TypeSafeActivities(mock_client())
    system_one = _activity(support)
    result = await system_one(
        SystemOneInput(
            state="state",
            questions={"billing": {"type": "noul", "instructions": "yes?"}},
            response_model="billing-activity",
        )
    )
    assert result["billing"] == {"type": "noul", "noul": 0.8}
    assert result["model"] == "jev-1.13.0"


@pytest.mark.asyncio
async def test_unregistered_response_model_name_is_non_retryable() -> None:
    """The registry lookup sits inside the translation block on purpose."""
    support, _ = _activities()
    system_one = _activity(support)
    with pytest.raises(ApplicationError) as err:
        await system_one(
            SystemOneInput(
                state="state",
                questions={"n": {"type": "noul", "instructions": "Is this spam?"}},
                response_model="never-registered",
            )
        )
    assert err.value.non_retryable is True
    assert err.value.type == "TypeSafeClientError"


@pytest.mark.asyncio
async def test_activity_preserves_sdk_client_http_timeout() -> None:
    """Configured HTTP phases reach the transport without an Activity override."""
    timeouts: list[dict[str, float | None]] = []

    def transport(request: httpx2.Request) -> httpx2.Response:
        timeouts.append(request.extensions["timeout"])
        return fake_response(
            body={
                "model": "jev-1.13.0",
                "answers": {"n": {"type": "noul", "noul": 0.8}},
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        )

    environment = ActivityEnvironment()
    environment.info = fake_info(timedelta(seconds=5))
    async with AsyncTypeSafeClient(
        api_key="fake",
        timeout=httpx2.Timeout(7, connect=0.25),
        retry=RetryPolicy(max_retries=0),
        transport=httpx2.MockTransport(transport),
    ) as client:
        await environment.run(
            TypeSafeActivities(client).system_one_activity,
            SystemOneInput(
                state="state",
                questions={"n": {"type": "noul", "instructions": "Is this spam?"}},
            ),
        )
    assert timeouts == [{"connect": 0.25, "read": 7, "write": 7, "pool": 7}]


@pytest.mark.asyncio
async def test_request_id_captured_from_protocol_header() -> None:
    client = mock_client(headers={"x-typesafe-request-id": "req_protocol"})
    assert await _system_one_request_id(client) == "req_protocol"


@pytest.mark.asyncio
async def test_request_id_falls_back_to_standard_header() -> None:
    client = mock_client(headers={"x-request-id": "req_standard"})
    assert await _system_one_request_id(client) == "req_standard"


@pytest.mark.asyncio
async def test_request_id_prefers_protocol_over_standard_header() -> None:
    client = mock_client(
        headers={
            "x-typesafe-request-id": "req_protocol",
            "x-request-id": "req_standard",
        }
    )
    assert await _system_one_request_id(client) == "req_protocol"


@pytest.mark.asyncio
async def test_request_id_absent_is_none() -> None:
    assert await _system_one_request_id(mock_client()) is None
