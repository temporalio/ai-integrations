"""LangSmith plugin for Temporal SDK."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.metadata
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import langchain_core.env
import langsmith

# langsmith conditionally imports langchain_core when it is installed.
# Pre-import the lazily-loaded submodule so it is in sys.modules before the
# workflow sandbox starts; otherwise the sandbox's __getattr__-triggered
# import hits restrictions on concurrent.futures.ThreadPoolExecutor.
try:
    import langchain_core.runnables.config  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError:
    pass

from temporalio.langsmith._interceptor import LangSmithInterceptor
from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

if TYPE_CHECKING:
    from langsmith.address import AgentAddress


class LangSmithPlugin(SimplePlugin):
    """LangSmith tracing plugin for Temporal SDK.

    .. warning::
        This class is experimental and may change in future versions.
        Use with caution in production environments.

    Provides automatic LangSmith run creation for workflows, activities,
    and other Temporal operations with context propagation.
    """

    def __init__(
        self,
        *,
        client: langsmith.Client | None = None,
        project_name: str | None = None,
        address: AgentAddress | None = None,
        add_temporal_runs: bool = False,
        default_metadata: dict[str, Any] | None = None,
        default_tags: list[str] | None = None,
    ) -> None:
        """Initialize the LangSmith plugin.

        Args:
            client: A langsmith.Client instance. If None, one will be created
                automatically (using LANGSMITH_API_KEY env var).
            project_name: Fallback LangSmith project name for traces with no
                parent. Temporal runs inherit their parent's destination when
                present. When unset, root runs use LangSmith's environment defaults.
            address: Fallback LangSmith agent/environment address for traces with
                no parent, such as ``langsmith.agent_address("support", "staging")``.
                Mutually exclusive with project_name. Requires LangSmith 0.14.10
                or later.
            add_temporal_runs: Whether to create LangSmith runs for Temporal
                operations. Defaults to False.
            default_metadata: Default metadata to attach to all runs.
            default_tags: Default tags to attach to all runs.
        """
        interceptor = LangSmithInterceptor(
            client=client,
            project_name=project_name,
            address=address,
            add_temporal_runs=add_temporal_runs,
            default_metadata=default_metadata,
            default_tags=default_tags,
        )
        interceptors = [interceptor]

        def workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner:
            if not runner:
                raise ValueError("No WorkflowRunner provided to the LangSmith plugin.")
            if isinstance(runner, SandboxedWorkflowRunner):
                # LangChain's first trace reads the platform, which can start a
                # subprocess the sandbox blocks. Read and cache it now.
                langchain_core.env.get_runtime_environment()
                return dataclasses.replace(
                    runner,
                    restrictions=runner.restrictions.with_passthrough_modules(
                        "langsmith",
                        "langchain_core",
                        "opentelemetry",
                    ),
                )
            return runner

        @asynccontextmanager
        async def run_context() -> AsyncGenerator[None, None]:
            try:
                yield
            finally:
                # Queued on the write thread, so it runs after every pending write.
                await asyncio.wrap_future(
                    interceptor._executor.submit(interceptor._client.flush)
                )

        super().__init__(
            "langchain.LangSmithPlugin",
            interceptors=interceptors,
            workflow_runner=workflow_runner,
            run_context=run_context,
        )

    def version(self) -> str:
        """Return this package's version, which worker heartbeats report."""
        try:
            return importlib.metadata.version("temporalio-langsmith")
        except importlib.metadata.PackageNotFoundError:
            return ""
