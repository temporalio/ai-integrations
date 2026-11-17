"""The published ranges stop below the next LangGraph and LangChain Core majors.

The plugin imports private LangGraph modules and recognizes a LangChain Core
function by its name and module. A new major can move any of them. The metadata
of a published release cannot be changed, so a range must not admit a major the
suite has not run on.
"""

from __future__ import annotations

import importlib.metadata

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version


def _published_range(package: str) -> SpecifierSet:
    for line in importlib.metadata.requires("temporalio-langgraph") or []:
        requirement = Requirement(line)
        if canonicalize_name(requirement.name) == package:
            return requirement.specifier
    raise AssertionError(f"temporalio-langgraph does not depend on {package}")


@pytest.mark.parametrize("package", ["langgraph", "langchain-core"])
def test_range_stops_below_the_next_major(package: str) -> None:
    installed = Version(importlib.metadata.version(package))
    next_major = f"{installed.major + 1}.0.0"
    published = _published_range(package)
    assert not published.contains(next_major), (
        f"temporalio-langgraph requires {package}{published}, which admits "
        f"{package} {next_major}. Set an upper bound below {next_major} in "
        "pyproject.toml."
    )
