"""Run LangChain's sync calls directly in the workflow.

LangChain runs a sync tool, output parser, ``RunnableLambda`` or LangGraph step
from async code with its ``run_in_executor`` helper, which runs it on a thread.
The workflow event loop has no threads, so the call raises
``NotImplementedError``. The modules that make those calls import the helper by
name and pass through the workflow sandbox, so the plugin rebinds that name to
a stand-in that runs the call inline in a workflow and calls the helper
everywhere else. The helper's other callers, such as caches, retrievers, vector
stores and ``ShellToolMiddleware``, do I/O and fail as before.

LangGraph stores the helper in each sync step when it builds a graph, so a
graph built before the Worker or Replayer keeps it. Workflow code is imported
again in the sandbox for each run, so its graphs get the stand-in. LangGraph's
refusal of a timeout on a sync step recognizes both.

Sync graph calls use a separate thread pool. Guard its module-local factory so
they fail before any node runs, including through methods bound before setup.
"""

from __future__ import annotations

import contextvars
import importlib
import threading
from collections.abc import Callable
from functools import partial
from typing import Any

from temporalio import workflow

# The modules whose ``run_in_executor`` runs a tool, an output parser, a
# runnable with only a sync invoke, or a LangGraph step.
_MODULES = (
    "langchain_core.tools.base",
    "langchain_core.tools.simple",
    "langchain_core.tools.structured",
    "langchain_core.output_parsers.base",
    "langchain_core.output_parsers.transform",
    "langchain_core.runnables.base",
    "langgraph._internal._runnable",
    "langgraph.pregel._call",
)

# From LangGraph 1.2.0, this module compares a sync step's function with the
# helper, to refuse a timeout on that step. Steps built before the swap hold
# the helper and later ones the stand-in, so the plugin wraps the check.
_TIMEOUT_CHECK = "langgraph.pregel._utils"
_EXECUTOR_CHECK = "_is_executor_backed_afunc"
_GRAPH_EXECUTOR = "langgraph.pregel._executor"


def _get_executor_for_config(config: Any) -> Any:
    if workflow.in_workflow() or workflow.unsafe.in_sandbox():
        from temporalio.deepagents.workflow import DeepAgentsWorkflowError

        raise DeepAgentsWorkflowError(
            "LangGraph sync graph calls start threads outside Workflow execution. "
            "Use await graph.ainvoke(...), graph.astream(...), or an Activity."
        )
    from langchain_core.runnables.config import get_executor_for_config

    return get_executor_for_config(config)


async def _run_in_executor(
    executor_or_config: Any, func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    if not workflow.in_workflow():
        from langchain_core.runnables.config import run_in_executor

        return await run_in_executor(executor_or_config, func, *args, **kwargs)
    try:
        # Like the helper, run it in a copy of the current context.
        return contextvars.copy_context().run(func, *args, **kwargs)
    except StopIteration as exc:
        raise RuntimeError from exc


def _recognize_stand_in(check: Callable[..., Any], helper: Any) -> Any:
    def is_executor_backed(afunc: Any) -> bool:
        if isinstance(afunc, partial) and afunc.func is _run_in_executor:
            afunc = partial(helper, *afunc.args, **afunc.keywords)
        return bool(check(afunc))

    is_executor_backed.recognizes_stand_in = True  # type: ignore[attr-defined]
    return is_executor_backed


_install_lock = threading.Lock()


def run_sync_calls_in_workflows() -> None:
    """Rebind the helper in ``_MODULES`` for both Workers and Replayers."""
    from langchain_core.runnables.config import get_executor_for_config, run_in_executor

    with _install_lock:
        modules = [importlib.import_module(name) for name in _MODULES]
        for module in modules:
            current = getattr(module, "run_in_executor", None)
            if current is not run_in_executor and current is not _run_in_executor:
                raise RuntimeError(
                    f"Unsupported LangChain run_in_executor in {module.__name__}; "
                    "the Deep Agents plugin requires the helper imported by name "
                    "to run sync tools in the workflow."
                )
        executor = importlib.import_module(_GRAPH_EXECUTOR)
        factory = getattr(executor, "get_executor_for_config", None)
        if (
            factory is not get_executor_for_config
            and factory is not _get_executor_for_config
        ):
            raise RuntimeError(
                f"Unsupported LangGraph executor factory in {_GRAPH_EXECUTOR}; "
                "the Deep Agents plugin requires get_executor_for_config imported by name."
            )
        if (
            "get_executor_for_config"
            not in executor.BackgroundExecutor.__init__.__code__.co_names
        ):
            raise RuntimeError(
                f"Unsupported LangGraph BackgroundExecutor in {_GRAPH_EXECUTOR}; "
                "it no longer uses get_executor_for_config."
            )
        utils = importlib.import_module(_TIMEOUT_CHECK)
        check = None
        if getattr(utils, "run_in_executor", None) is run_in_executor:
            check = getattr(utils, _EXECUTOR_CHECK, None)
            if not callable(check):
                raise RuntimeError(
                    f"Unsupported LangGraph timeout check in {_TIMEOUT_CHECK}; "
                    "the Deep Agents plugin requires its "
                    f"{_EXECUTOR_CHECK} to keep refusing timeouts on sync steps."
                )
        for module in modules:
            setattr(module, "run_in_executor", _run_in_executor)
        setattr(executor, "get_executor_for_config", _get_executor_for_config)
        if check is not None and not getattr(check, "recognizes_stand_in", False):
            wrapped = _recognize_stand_in(check, run_in_executor)
            setattr(utils, _EXECUTOR_CHECK, wrapped)
