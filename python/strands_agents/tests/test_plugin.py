import typing
from dataclasses import replace

import pytest
from botocore.config import Config as BotocoreConfig
from strands.interrupt import Interrupt, InterruptException
from strands.models import Model
from strands.models.bedrock import DEFAULT_READ_TIMEOUT

import temporalio.strands_agents._plugin as plugin_module
from temporalio.api.failure.v1 import Failure
from temporalio.client import ClientConfig
from temporalio.contrib.pydantic import PydanticPayloadConverter
from temporalio.converter import (
    DataConverter,
    DefaultFailureConverter,
    DefaultFailureConverterWithEncodedAttributes,
    PayloadConverter,
)
from temporalio.exceptions import ApplicationError
from temporalio.strands_agents import StrandsPlugin
from temporalio.strands_agents._failure_converter import (
    StrandsFailureConverter,
    StrandsFailureConverterWithEncodedAttributes,
)
from temporalio.strands_agents._model_activity import ModelActivity
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner
from tests.common import CompressionCodec
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


class DomainError(Exception):
    pass


class DomainFailureConverter(DefaultFailureConverter):
    def from_failure(
        self, failure: Failure, payload_converter: PayloadConverter
    ) -> BaseException:
        if failure.application_failure_info.type == "DomainError":
            return DomainError(failure.message)
        return super().from_failure(failure, payload_converter)


class CallerEncodedFailureConverter(DefaultFailureConverterWithEncodedAttributes):
    pass


def _configure(target: str, supplied: DataConverter) -> DataConverter:
    plugin = StrandsPlugin(models={})
    configured: DataConverter | None
    if target == "client":
        configured = plugin.configure_client(_client_config(supplied))["data_converter"]
    elif target == "replayer":
        configured = plugin.configure_replayer(
            {"data_converter": supplied, "workflow_runner": SandboxedWorkflowRunner()}
        ).get("data_converter")
    else:
        raise AssertionError(target)
    assert configured is not None
    return configured


@pytest.mark.parametrize("target", ["client", "replayer"])
@pytest.mark.parametrize(
    "failure_converter_class",
    [DefaultFailureConverterWithEncodedAttributes, CallerEncodedFailureConverter],
)
def test_caller_encoded_failure_converter_is_kept(
    target: str,
    failure_converter_class: type[DefaultFailureConverterWithEncodedAttributes],
) -> None:
    supplied = replace(
        DataConverter.default, failure_converter_class=failure_converter_class
    )
    configured = _configure(target, supplied)

    # The caller's encoding still applies to failure messages.
    failure = Failure()
    configured.failure_converter.to_failure(
        ApplicationError("secret"), configured.payload_converter, failure
    )
    assert failure.message == "Encoded failure"
    decoded = configured.failure_converter.from_failure(
        failure, configured.payload_converter
    )
    assert isinstance(decoded, ApplicationError)
    assert decoded.message == "secret"

    # The Strands handling is added on top of it.
    interrupt = Failure()
    configured.failure_converter.to_failure(
        InterruptException(Interrupt(id="i1", name="approval", reason="ok?")),
        configured.payload_converter,
        interrupt,
    )
    assert interrupt.application_failure_info.type == "StrandsInterrupt"
    assert interrupt.application_failure_info.non_retryable
    assert interrupt.message == "Encoded failure"

    assert (
        configured.failure_converter_class
        is StrandsFailureConverterWithEncodedAttributes
    )
    # Running the plugin again on its own output keeps it.
    assert _configure(target, configured).failure_converter_class is (
        StrandsFailureConverterWithEncodedAttributes
    )


@pytest.mark.parametrize("target", ["client", "replayer"])
def test_other_failure_converter_is_replaced(target: str) -> None:
    # Keeping this class would change which errors Workflow code reads, and so
    # break the replay of existing histories.
    supplied = replace(
        DataConverter.default, failure_converter_class=DomainFailureConverter
    )
    configured = _configure(target, supplied)
    assert configured.failure_converter_class is StrandsFailureConverter
    failure = Failure()
    configured.failure_converter.to_failure(
        ApplicationError("boom", type="DomainError"),
        configured.payload_converter,
        failure,
    )
    decoded = configured.failure_converter.from_failure(
        failure, configured.payload_converter
    )
    assert isinstance(decoded, ApplicationError)
    assert decoded.type == "DomainError"
