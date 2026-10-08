"""The worker plugin."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkerConfig

from ._activity import CodexActivities, ObserverFactory


class CodexPlugin(SimplePlugin):
    """Registers the Codex segment Activity (and holds the Codex configuration) on a Worker.

    Everything that must stay out of Workflow history lives here, on the Worker.

    Args:
        codex_bin: Path to the ``codex`` binary. Defaults to ``$CODEX_BIN`` or the binary bundled
            with the ``bundled-codex`` extra.
        config_overrides: ``--config key=value`` pairs passed to ``codex`` (for example a custom
            ``model_provider``). Per process, so they apply to every thread.
        env: Extra environment variables for the app-server process (for example
            ``OPENAI_API_KEY``). The Worker's own environment is inherited.
        home_root: Directory under which each segment gets a fresh ``CODEX_HOME``. Defaults to the
            system temp directory.
        auth_file: Path to a Codex login (``auth.json``, from ``codex login``, for example
            ``~/.codex/auth.json``) to use instead of an API key. Each segment gets a private copy in
            its fresh ``CODEX_HOME``, and refreshed tokens are written back to this file. Keep
            ``OPENAI_API_KEY`` out of the Worker's environment so the login is the one used.
        observer_factory: Builds a :class:`~temporalio.openai_codex.CodexObserver` (an async
            context manager) from a segment's ``observer_context``, to stream live events. Only
            consulted for segments run with an ``observer_context``.
    """

    def __init__(
        self,
        *,
        codex_bin: str | None = None,
        config_overrides: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        home_root: str | None = None,
        observer_factory: ObserverFactory | None = None,
        auth_file: str | None = None,
    ) -> None:
        """Create the plugin; see the class docstring for the arguments."""
        self._impl = CodexActivities(
            codex_bin=codex_bin,
            config_overrides=config_overrides,
            env=env,
            home_root=home_root,
            observer_factory=observer_factory,
            auth_file=auth_file,
        )

        def add_activities(
            existing: Sequence[Callable[..., Any]] | None,
        ) -> Sequence[Callable[..., Any]]:
            return [*(existing or []), self._impl.run_segment]

        super().__init__(name="CodexPlugin", activities=add_activities)

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        """Also hand the Worker's client to the Activity, which uses it to ask for approvals."""
        config = super().configure_worker(config)
        client = config.get("client")
        if client is not None:
            self._impl.bind_client(client)
        return config
