"""Data converter composition preserves codecs and compatible custom settings."""

import zlib
from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import pytest

from temporalio.api.common.v1 import Payload
from temporalio.client import ClientConfig
from temporalio.contrib.pydantic import PydanticPayloadConverter
from temporalio.converter import DataConverter, DefaultFailureConverter, PayloadCodec
from temporalio.deepagents import DeepAgentsPlugin
from temporalio.deepagents._serde import DeepAgentsPayloadConverter
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner


class CompressionCodec(PayloadCodec):
    async def encode(self, payloads: Sequence[Payload]) -> list[Payload]:
        return [
            Payload(
                metadata={"encoding": b"binary/zlib"},
                data=zlib.compress(payload.SerializeToString()),
            )
            for payload in payloads
        ]

    async def decode(self, payloads: Sequence[Payload]) -> list[Payload]:
        return [
            Payload.FromString(zlib.decompress(payload.data)) for payload in payloads
        ]


class CustomFailureConverter(DefaultFailureConverter):
    pass


class CustomPayloadConverter(DeepAgentsPayloadConverter):
    pass


def _client_config(converter: DataConverter | None = None) -> ClientConfig:
    # SimplePlugin reads only these fields; the rest belong to Client.connect.
    config = {"data_converter": converter} if converter is not None else {}
    return cast(ClientConfig, cast(object, config))


@pytest.mark.parametrize("target", ["client", "replayer"])
async def test_existing_codec_round_trip(target: str) -> None:
    codec = CompressionCodec()
    original = replace(
        DataConverter.default,
        payload_codec=codec,
        failure_converter_class=CustomFailureConverter,
    )
    value = {"message": "preserve the codec"}
    existing_payloads = await original.encode([value])
    plugin = DeepAgentsPlugin()
    configured: DataConverter | None
    if target == "client":
        configured = plugin.configure_client(_client_config(original))["data_converter"]
    else:
        configured = plugin.configure_replayer(
            {
                "data_converter": original,
                "workflow_runner": SandboxedWorkflowRunner(),
            }
        ).get("data_converter")

    assert configured is not None
    assert configured.payload_codec is codec
    assert configured.failure_converter_class is CustomFailureConverter
    assert configured.payload_converter_class is DeepAgentsPayloadConverter
    assert await configured.decode(existing_payloads) == [value]
    new_payloads = await configured.encode([value])
    assert new_payloads[0].metadata["encoding"] == b"binary/zlib"
    assert await original.decode(new_payloads) == [value]


@pytest.mark.parametrize(
    "payload_converter_class", [DeepAgentsPayloadConverter, CustomPayloadConverter]
)
@pytest.mark.parametrize("explicit", [False, True])
def test_compatible_converter(
    payload_converter_class: type[DeepAgentsPayloadConverter],
    explicit: bool,
) -> None:
    supplied = DataConverter(
        payload_converter_class=payload_converter_class,
        payload_codec=CompressionCodec(),
        failure_converter_class=CustomFailureConverter,
    )
    plugin = DeepAgentsPlugin(data_converter=supplied if explicit else None)
    configured = plugin.configure_client(
        _client_config(DataConverter.default if explicit else supplied)
    )["data_converter"]
    assert configured is supplied


def test_explicit_incompatible_converter_is_rejected() -> None:
    supplied = DataConverter(payload_converter_class=PydanticPayloadConverter)
    with pytest.raises(ValueError, match="cannot compose"):
        DeepAgentsPlugin(data_converter=supplied)


def test_existing_incompatible_converter_is_preserved_on_error() -> None:
    supplied = DataConverter(payload_converter_class=PydanticPayloadConverter)
    config = _client_config(supplied)
    with pytest.raises(ValueError, match="cannot compose"):
        DeepAgentsPlugin().configure_client(config)
    assert config["data_converter"] is supplied


def test_default_converter() -> None:
    configured = DeepAgentsPlugin().configure_client(_client_config())["data_converter"]
    assert configured.payload_converter_class is DeepAgentsPayloadConverter
