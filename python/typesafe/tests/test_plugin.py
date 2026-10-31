"""Plugin construction and caller-owned SDK client behavior."""

from __future__ import annotations

import pytest
from typesafe_sdk import RetryPolicy, SystemOneResponse, TypeSafeError

from temporalio.contrib.pydantic import PydanticPayloadConverter
from temporalio.converter import (
    DataConverter,
    DefaultFailureConverterWithEncodedAttributes,
    DefaultPayloadConverter,
    PayloadConverter,
)
from temporalio.typesafe._plugin import TypeSafePlugin, _data_converter
from temporalio.typesafe._types import lookup_response_model
from tests.helpers.codec import CompressionCodec
from tests.helpers.fake_typesafe import mock_client


class _CustomResponse(SystemOneResponse):
    pass


class _CustomPayloadConverter(DefaultPayloadConverter):
    pass


def test_plugin_accepts_caller_configured_sdk_client() -> None:
    client = mock_client(retry=RetryPolicy(max_retries=0))
    plugin = TypeSafePlugin(client)
    assert plugin._support._client is client


def test_plugin_rejects_sdk_retries() -> None:
    with pytest.raises(TypeSafeError, match="max_retries=0"):
        TypeSafePlugin(mock_client())
    with pytest.raises(TypeSafeError, match="max_retries=0"):
        TypeSafePlugin(mock_client(retry=RetryPolicy(max_retries=1)))
    TypeSafePlugin(mock_client(retry=RetryPolicy(max_retries=0)))


def test_plugin_registers_response_models() -> None:
    TypeSafePlugin(
        mock_client(retry=RetryPolicy(max_retries=0)),
        response_models={"plugin-registered": _CustomResponse},
    )
    assert lookup_response_model("plugin-registered") is _CustomResponse


def test_plugin_rejects_non_response_model() -> None:
    with pytest.raises(TypeSafeError, match="must subclass"):
        TypeSafePlugin(
            mock_client(retry=RetryPolicy(max_retries=0)),
            response_models={"bad": int},  # type: ignore[dict-item]
        )
    with pytest.raises(TypeSafeError, match="not registered"):
        lookup_response_model("bad")


def test_plugin_duplicate_name_with_other_class_rejects() -> None:
    TypeSafePlugin(
        mock_client(retry=RetryPolicy(max_retries=0)),
        response_models={"plugin-dup": _CustomResponse},
    )

    class Other(SystemOneResponse):
        pass

    with pytest.raises(TypeSafeError, match="already registered"):
        TypeSafePlugin(
            mock_client(retry=RetryPolicy(max_retries=0)),
            response_models={"plugin-dup": Other},
        )


def test_data_converter_upgrades_default() -> None:
    """Upgrading payload conversion preserves the codec and failure settings."""
    # The SDK may wrap the converter instance for transfer type conversion.
    assert _data_converter(None).payload_converter_class is PydanticPayloadConverter
    codec = CompressionCodec()
    plain = DataConverter(
        payload_codec=codec,
        failure_converter_class=DefaultFailureConverterWithEncodedAttributes,
    )
    assert plain.payload_converter_class is DefaultPayloadConverter
    upgraded = _data_converter(plain)
    assert upgraded is not plain
    assert upgraded.payload_converter_class is PydanticPayloadConverter
    assert upgraded.payload_codec is codec
    assert (
        upgraded.failure_converter_class is DefaultFailureConverterWithEncodedAttributes
    )
    assert plain.payload_converter_class is DefaultPayloadConverter


@pytest.mark.parametrize(
    "payload_converter_class", [PydanticPayloadConverter, _CustomPayloadConverter]
)
def test_custom_payload_converter_is_preserved(
    payload_converter_class: type[PayloadConverter],
) -> None:
    custom = DataConverter(
        payload_converter_class=payload_converter_class,
        payload_codec=CompressionCodec(),
        failure_converter_class=DefaultFailureConverterWithEncodedAttributes,
    )
    assert _data_converter(custom) is custom
