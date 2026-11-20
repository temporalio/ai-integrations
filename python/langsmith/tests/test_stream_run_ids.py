"""LangSmith cooperates with a workflow-local streaming ID provider."""

from uuid import UUID

import pytest

from temporalio import workflow
from temporalio.langsmith import _run_tree


@pytest.mark.parametrize("provided", [None, UUID(int=42)])
def test_langchain_run_ids_honor_stream_provider(
    monkeypatch: pytest.MonkeyPatch, provided: UUID | None
) -> None:
    monkeypatch.setattr(workflow, "in_workflow", lambda: True)
    monkeypatch.setattr(
        workflow,
        "extern_functions",
        lambda: {"temporalio.langchain.run_id": lambda: provided},
    )
    monkeypatch.setattr(_run_tree, "run_id", lambda: UUID(int=17))
    assert _run_tree._langchain_run_id() == (provided or UUID(int=17))


def test_langchain_run_ids_without_stream_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow, "in_workflow", lambda: True)
    monkeypatch.setattr(workflow, "extern_functions", lambda: {})
    monkeypatch.setattr(_run_tree, "run_id", lambda: UUID(int=17))
    assert _run_tree._langchain_run_id() == UUID(int=17)
