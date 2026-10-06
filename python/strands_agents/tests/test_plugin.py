import typing
import zlib
from collections.abc import Sequence
from dataclasses import replace

import pytest
from botocore.config import Config as BotocoreConfig
from strands.models import Model
from strands.models.bedrock import DEFAULT_READ_TIMEOUT

import temporalio.strands_agents._plugin as plugin_module
from temporalio.api.common.v1 import Payload
from temporalio.client import ClientConfig
from temporalio.contrib.pydantic import PydanticPayloadConverter
from temporalio.converter import DataConverter, PayloadCodec
from temporalio.strands_agents import StrandsPlugin
from temporalio.strands_agents._failure_converter import StrandsFailureConverter
from temporalio.strands_agents._model_activity import ModelActivity
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner
from tests.mock_model import MockModel


def test_default_bedrock_model_disables_botocore_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_configs: list[BotocoreConfig] = []
    model = MockModel([])

    def bedrock_model(*, boto_client_config: BotocoreConfig) -> Model:
        captured_configs.append(boto_client_config)
        return model

    monkeypatch.setattr(plugin_module, "BedrockModel", bedrock_model)

    plugin = StrandsPlugin()
    activities = plugin.activities
    assert isinstance(activities, list)
    model_activity = typing.cast(
        ModelActivity,
        typing.cast(typing.Any, activities[0]).__self__,
    )

    assert model_activity._get_model(None) is model
    assert len(captured_configs) == 1
    assert getattr(captured_configs[0], "read_timeout") == DEFAULT_READ_TIMEOUT
    assert getattr(captured_configs[0], "retries") == {"max_attempts": 0}


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


def _client_config(converter: DataConverter | None = None) -> ClientConfig:
    # SimplePlugin reads only these fields; the rest belong to Client.connect.
    config = {"data_converter": converter} if converter is not None else {}
    return typing.cast(ClientConfig, typing.cast(object, config))


@pytest.mark.parametrize("target", ["client", "replayer"])
async def test_existing_codec_round_trip(target: str) -> None:
    codec = CompressionCodec()
    original = replace(DataConverter.default, payload_codec=codec)
    value = {"message": "preserve the codec"}
    existing_payloads = await original.encode([value])
    plugin = StrandsPlugin(models={})
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
    assert configured.payload_converter_class is PydanticPayloadConverter
    assert configured.failure_converter_class is StrandsFailureConverter
    assert await configured.decode(existing_payloads) == [value]
    new_payloads = await configured.encode([value])
    assert new_payloads[0].metadata["encoding"] == b"binary/zlib"
    assert await original.decode(new_payloads) == [value]


def test_default_converter() -> None:
    configured = StrandsPlugin(models={}).configure_client(_client_config())[
        "data_converter"
    ]
    assert configured.payload_converter_class is PydanticPayloadConverter
    assert configured.failure_converter_class is StrandsFailureConverter


def test_custom_payload_converter_is_preserved() -> None:
    supplied = DataConverter(
        payload_converter_class=PydanticPayloadConverter,
        payload_codec=CompressionCodec(),
    )
    configured = StrandsPlugin(models={}).configure_client(_client_config(supplied))[
        "data_converter"
    ]
    assert configured is supplied
