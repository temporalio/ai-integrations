"""An example agent built on the plugin: a coding assistant with three roles.

One module per concern. The workflow is the manager: each role is one Claude turn, and the
workflow hands one role's typed output to the next.

* ``manager.py``: the workflow, which hands each message from planner to coder to reviewer.
* ``planner.py``, ``coder.py``, ``reviewer.py``: each role's prompt, options, tools and output.
* ``policy.py``: the coder's approval policy for native calls, an allowlist.
* ``tools.py``: the workflow tools; ``tracker.py``: the Activities they reach the tracker with.

The offline scenarios are ``tests/test_claude_coding_assistant.py``; the same workflow runs
against the real CLI in ``tests/test_claude_real_cli.py``.
"""
