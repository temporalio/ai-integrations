"""The ``bytes`` codec touches the fields Strands types as ``bytes`` and nothing else."""

import base64
import copy
from typing import Any

import pytest

from temporalio.strands_agents._bytes_values import (
    decode_event,
    decode_messages,
    decode_tool_result,
    encode_event,
    encode_messages,
    encode_tool_result,
)

DATA = b"\xff\x00\xfebytes"
TAGGED = {"__bytes_encoded__": True, "data": base64.b64encode(DATA).decode()}


def _message_with_every_bytes_field() -> list[Any]:
    source = {"source": {"bytes": DATA}}
    return [
        {
            "role": "user",
            "content": [
                {"image": {"format": "png", **source}},
                {"document": {"format": "pdf", "name": "d", **source}},
                {"video": {"format": "mp4", **source}},
                {"audio": {"format": "wav", **source}},
                {"reasoningContent": {"redactedContent": DATA}},
                {
                    "toolResult": {
                        "toolUseId": "t",
                        "status": "success",
                        "content": [
                            {"image": {"format": "png", **source}},
                            {"document": {"format": "pdf", "name": "d", **source}},
                        ],
                    }
                },
            ],
        }
    ]


def test_every_bytes_field_round_trips_through_tagged_form():
    messages = _message_with_every_bytes_field()
    original = copy.deepcopy(messages)

    encoded = encode_messages(messages)

    assert messages == original
    blocks = encoded[0]["content"]
    assert blocks[0]["image"]["source"]["bytes"] == TAGGED
    assert blocks[3]["audio"]["source"]["bytes"] == TAGGED
    assert blocks[4]["reasoningContent"]["redactedContent"] == TAGGED
    assert blocks[5]["toolResult"]["content"][1]["document"]["source"]["bytes"] == (
        TAGGED
    )
    assert decode_messages(encoded) == original


def test_tool_result_round_trips_and_keeps_other_keys():
    result = {
        "toolUseId": "t",
        "status": "success",
        "content": [{"image": {"format": "png", "source": {"bytes": DATA}}}],
        "structuredContent": {"a": [1, 2]},
        "metadata": {"b": None},
        "isError": False,
    }

    encoded = encode_tool_result(result)

    assert encoded["content"][0]["image"]["source"]["bytes"] == TAGGED
    assert {k: v for k, v in encoded.items() if k != "content"} == {
        k: v for k, v in result.items() if k != "content"
    }
    assert decode_tool_result(encoded) == result


def test_event_round_trips():
    event = {
        "contentBlockDelta": {"delta": {"reasoningContent": {"redactedContent": DATA}}}
    }

    encoded = encode_event(event)

    assert encoded["contentBlockDelta"]["delta"]["reasoningContent"][
        "redactedContent"
    ] == (TAGGED)
    assert decode_event(encoded) == event


@pytest.mark.parametrize(
    "recorded",
    ["ASCII-redacted", [65, 83, 67, 73, 73, 45, 114, 101, 100, 97, 99, 116, 101, 100]],
    ids=["pydantic_string", "default_converter_ints"],
)
def test_values_recorded_before_tagging_decode_to_bytes(recorded: Any):
    """A string or list of ints in a ``bytes`` field is what older histories hold."""
    event = {
        "contentBlockDelta": {
            "delta": {"reasoningContent": {"redactedContent": recorded}}
        }
    }

    decoded = decode_event(event)

    assert decoded["contentBlockDelta"]["delta"]["reasoningContent"] == {
        "redactedContent": b"ASCII-redacted"
    }
    result = {"content": [{"image": {"format": "png", "source": {"bytes": recorded}}}]}
    assert decode_tool_result(result)["content"][0]["image"]["source"] == {
        "bytes": b"ASCII-redacted"
    }


@pytest.mark.parametrize("junk", [None, 5, [], "", "x", ["a"]])
def test_other_values_do_not_raise(junk: Any):
    """Malformed input is passed along rather than failing the workflow task."""
    assert encode_messages(junk) == junk
    assert decode_messages(junk) == junk
    assert decode_event(junk) == junk
    assert decode_tool_result(junk) == junk
