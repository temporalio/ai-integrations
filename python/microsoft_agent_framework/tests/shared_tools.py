"""A tool module that workflow files import through the sandbox passthrough.

The sandbox re-imports ordinary modules per workflow run, so a tool defined
here and imported under ``workflow.unsafe.imports_passed_through()`` is one
object shared by every workflow instance in the worker.
"""

import threading

from agent_framework import tool


@tool
def shared_sync_weather(city: str) -> str:
    """Report the weather for a city from a synchronous tool."""
    return f"sunny in {city}"


@tool
def shared_sync_thread(label: str) -> str:
    """Report the thread a synchronous tool ran on."""
    return f"{label}:{threading.get_ident()}"
