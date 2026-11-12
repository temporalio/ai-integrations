"""Generated dataclass signatures are not model-facing tool descriptions."""

import sys
from dataclasses import dataclass

import pytest

from temporalio import activity
from temporalio.google_genai.workflow import activity_as_tool


@pytest.mark.skipif(sys.version_info < (3, 14), reason="requires deferred annotations")
def test_dataclass_tool_omits_generated_docstring() -> None:
    """A forward reference resolved after class creation stays out of the description."""

    @dataclass
    class UndocumentedActivity:
        state: State  # type: ignore

        async def __call__(self, value: str) -> str:
            return value

    class State:  # type: ignore[reportUnusedClass]
        pass

    tool = activity_as_tool(activity.defn(UndocumentedActivity))
    assert tool.__doc__ is None
