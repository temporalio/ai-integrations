import importlib.metadata
import typing
from dataclasses import replace
from datetime import timedelta

import mcp
import pytest
from botocore.config import Config as BotocoreConfig
from strands.interrupt import Interrupt, InterruptException
from strands.models import Model
from strands.models.bedrock import DEFAULT_READ_TIMEOUT
from strands.tools.mcp import MCPClient as StrandsMCPClient

import temporalio.mcp
import temporalio.strands_agents._plugin as plugin_module
from temporalio.api.failure.v1 import Failure
from temporalio.client import ClientConfig
from temporalio.contrib.pydantic import (
    PydanticPayloadConverter,
    pydantic_data_converter,
)
from temporalio.converter import (
    DataConverter,
    DefaultFailureConverter,
    DefaultFailureConverterWithEncodedAttributes,
    FailureConverter,
    PayloadConverter,
)
from temporalio.exceptions import ApplicationError
from temporalio.strands_agents import StrandsPlugin
from temporalio.strands_agents._failure_converter import (
    StrandsFailureConverter,
    StrandsFailureConverterWithEncodedAttributes,
)
from temporalio.strands_agents._model_activity import ModelActivity
from temporalio.testing import ActivityEnvironment
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
    assert configured.payload_converter_class is PydanticPayloadConverter
    assert configured.payload_codec is supplied.payload_codec


@pytest.mark.parametrize("target", ["client", "replayer"])
def test_custom_payload_converter_gets_strands_failure_converter(target: str) -> None:
    supplied = pydantic_data_converter
    configured = _configure(target, supplied)
    assert configured.payload_converter_class is supplied.payload_converter_class
    assert configured.failure_converter_class is StrandsFailureConverter
    failure = Failure()
    configured.failure_converter.to_failure(
        InterruptException(Interrupt(id="i1", name="approval", reason="ok?")),
        configured.payload_converter,
        failure,
    )
    assert failure.application_failure_info.type == "StrandsInterrupt"
    assert failure.application_failure_info.non_retryable


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


@pytest.mark.parametrize("target", ["client", "replayer"])
def test_custom_payload_converter_keeps_caller_failure_decoding(target: str) -> None:
    # With a non-default payload converter the caller's failure converter was
    # always preserved; its decoding must still run for recorded failures.
    supplied = replace(
        pydantic_data_converter, failure_converter_class=DomainFailureConverter
    )
    configured = _configure(target, supplied)
    failure = Failure()
    configured.failure_converter.to_failure(
        ApplicationError("boom", type="DomainError"),
        configured.payload_converter,
        failure,
    )
    decoded = configured.failure_converter.from_failure(
        failure, configured.payload_converter
    )
    assert isinstance(decoded, DomainError)
    # Strands handling is still added on top.
    interrupt = Failure()
    configured.failure_converter.to_failure(
        InterruptException(Interrupt(id="i1", name="approval", reason="ok?")),
        configured.payload_converter,
        interrupt,
    )
    assert interrupt.application_failure_info.type == "StrandsInterrupt"
    assert interrupt.application_failure_info.non_retryable
    # Running the plugin again on its own output keeps it.
    again = _configure(target, configured)
    assert isinstance(
        again.failure_converter.from_failure(failure, again.payload_converter),
        DomainError,
    )


class MinimalFailureConverter(FailureConverter):
    def to_failure(
        self,
        exception: BaseException,
        payload_converter: PayloadConverter,
        failure: Failure,
    ) -> None:
        failure.message = str(exception)
        failure.application_failure_info.type = type(exception).__name__

    def from_failure(
        self, failure: Failure, payload_converter: PayloadConverter
    ) -> BaseException:
        return DomainError(failure.message)


@pytest.mark.parametrize("target", ["client", "replayer"])
def test_minimal_failure_converter_with_custom_payload_converter_is_kept(
    target: str,
) -> None:
    supplied = replace(
        pydantic_data_converter, failure_converter_class=MinimalFailureConverter
    )
    configured = _configure(target, supplied)
    assert configured.failure_converter_class is MinimalFailureConverter
    decoded = configured.failure_converter.from_failure(
        Failure(message="x"), configured.payload_converter
    )
    assert isinstance(decoded, DomainError)


def _old_strands_client() -> typing.Any:
    # A factory left as it was for Strands' MCPClient, the common upgrade mistake.
    return StrandsMCPClient(typing.cast(typing.Any, lambda: None))


def _client_on_a_function() -> mcp.Client:
    # The same factory with only its import changed to mcp.Client.
    return mcp.Client(typing.cast(typing.Any, lambda: None))


@pytest.mark.parametrize(
    ("factory", "message", "error_type"),
    [
        (
            _old_strands_client,
            "MCP server factory 'old' returned strands.tools.mcp.mcp_client.MCPClient, "
            "not mcp.Client; mcp_clients factories must return an mcp.Client "
            "(see the upgrade notes in the plugin README)",
            "MCPServerFactoryTypeError",
        ),
        (
            _client_on_a_function,
            "MCP server factory 'old' returned an mcp.Client whose server is a "
            "function; pass the transport itself, a StdioServerParameters or a URL, "
            "not a function that returns the transport "
            "(see the upgrade notes in the plugin README)",
            "MCPServerFactoryTransportError",
        ),
    ],
)
async def test_mcp_factory_that_returns_the_wrong_client_is_non_retryable(
    factory: typing.Callable[[], typing.Any], message: str, error_type: str
) -> None:
    plugin = StrandsPlugin(models={}, mcp_clients={"old": factory})
    activities = typing.cast(list[typing.Any], plugin.activities)
    list_tools = next(
        a
        for a in activities
        if a.__temporal_activity_definition.name == "temporalio.mcp.old.list-tools"
    )

    with pytest.raises(ApplicationError) as error:
        await ActivityEnvironment().run(list_tools, {"factory_argument": None})

    assert error.value.message == message
    assert error.value.type == error_type
    assert error.value.non_retryable


def test_plugin_reports_package_version() -> None:
    assert StrandsPlugin().version() == importlib.metadata.version(
        "temporalio-strands-agents"
    )


def test_mcp_factory_that_needs_arguments_fails_when_the_plugin_is_built() -> None:
    def tenant_factory(tenant: str) -> mcp.Client:
        return mcp.Client(f"https://{tenant}.example.com/mcp")

    with pytest.raises(
        TypeError, match="MCP server factory 'weather' must take no arguments"
    ):
        StrandsPlugin(
            models={}, mcp_clients={"weather": typing.cast(typing.Any, tenant_factory)}
        )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, timedelta(minutes=5)),
        ({"mcp_connection_idle_timeout": None}, None),
        ({"mcp_connection_idle_timeout": timedelta(0)}, timedelta(0)),
    ],
)
def test_mcp_connection_idle_timeout_is_passed_through(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, typing.Any],
    expected: timedelta | None,
) -> None:
    seen: list[timedelta | None] = []

    class RecordingMCPPlugin(temporalio.mcp.MCPPlugin):
        def __init__(self, clients: typing.Any, **options: typing.Any) -> None:
            seen.append(options["connection_idle_timeout"])
            super().__init__(clients, **options)

    monkeypatch.setattr(temporalio.mcp, "MCPPlugin", RecordingMCPPlugin)

    StrandsPlugin(
        models={},
        mcp_clients={"echo": lambda: mcp.Client("http://localhost")},
        **kwargs,
    )

    assert seen == [expected]
