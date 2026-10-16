"""Workflow tools: Claude calls them, and they run in the workflow.

Each reaches the tracker through an Activity, so its I/O is retried and recorded. A call that a
resumed attempt reissues with the same input is answered from that record, not run again.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from .tracker import TicketTracker

_TIMEOUT = timedelta(seconds=30)


async def lookup_ticket(ticket_id: str) -> str:
    """Read a ticket from the team's issue tracker by its id, such as ENG-42."""
    return await workflow.execute_activity_method(
        TicketTracker.fetch_ticket, ticket_id, start_to_close_timeout=_TIMEOUT
    )


async def comment_on_ticket(ticket_id: str, comment: str) -> str:
    """Post a comment on a ticket, such as what was changed to fix it."""
    await workflow.execute_activity_method(
        TicketTracker.add_comment,
        args=[ticket_id, comment],
        start_to_close_timeout=_TIMEOUT,
    )
    return "posted"
