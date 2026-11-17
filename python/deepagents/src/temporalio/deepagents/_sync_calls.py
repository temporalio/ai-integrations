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

LangChain runs a graph step with several nodes, ``batch`` and
``RunnableParallel`` on its own thread pool, whose threads run outside the
Workflow. The plugin swaps that pool's class for one that refuses work in
Workflow code, so those calls fail the Workflow Task before anything runs on a
thread. A graph whose steps each run one node never uses the pool.
"""

from __future__ import annotations

import contextvars
import importlib
import inspect
import threading
from collections.abc import Callable
from functools import partial
from typing import Any

from temporalio import workflow
from temporalio.worker.workflow_sandbox import RestrictedWorkflowAccessError

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

# Runs recorded before this change keep their old path: sync calls fail as
# before, and the thread pool stays open to them.
_PATCH = "deepagents-sync-calls-inline"

_POOL_MESSAGE = (
    "LangChain tried to run work on its thread pool in a Workflow, for example a "
    "LangGraph step with several nodes, batch() or RunnableParallel. Those threads "
    "run outside the Workflow. Use the async form, such as await graph.ainvoke(...), "
    "or move the call into an Activity."
)


def _refuse_in_workflows(pool: type) -> type:
    class WorkflowPool(pool):  # type: ignore[valid-type, misc]
        refuses_workflow_work = True

        def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
            if (
                workflow.patched(_PATCH)
                if workflow.in_workflow()
                else workflow.unsafe.in_sandbox()
            ):
                raise RestrictedWorkflowAccessError(
                    "langchain_core.runnables.config.ContextThreadPoolExecutor",
                    override_message=_POOL_MESSAGE,
                )
            return super().submit(fn, *args, **kwargs)

    return WorkflowPool


async def _run_in_executor(
    executor_or_config: Any, func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    if not workflow.in_workflow() or not workflow.patched(_PATCH):
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
    from langchain_core.runnables import config
    from langchain_core.runnables.config import run_in_executor

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
        factory = inspect.unwrap(config.get_executor_for_config)
        if "ContextThreadPoolExecutor" not in factory.__code__.co_names:
            raise RuntimeError(
                "Unsupported langchain_core.runnables.config.get_executor_for_config; "
                "the Deep Agents plugin requires it to build ContextThreadPoolExecutor."
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
        pool = config.ContextThreadPoolExecutor
        if not getattr(pool, "refuses_workflow_work", False):
            setattr(config, "ContextThreadPoolExecutor", _refuse_in_workflows(pool))
        if check is not None and not getattr(check, "recognizes_stand_in", False):
            wrapped = _recognize_stand_in(check, run_in_executor)
            setattr(utils, _EXECUTOR_CHECK, wrapped)
