"""The team's issue tracker, reached from the workflow through Activities."""

from __future__ import annotations

import json
import re
from pathlib import Path

from temporalio import activity
from temporalio.exceptions import ApplicationError

_TICKET_ID = re.compile(r"^[A-Z]+-[0-9]+$")


class TicketTracker:
    """One JSON file per ticket under ``root``, and its comments as JSON lines beside it."""

    def __init__(self, root: Path) -> None:
        """Serve the tickets under ``root``."""
        self._root = root

    @activity.defn
    async def fetch_ticket(self, ticket_id: str) -> str:
        """The ticket's JSON."""
        return self._ticket(ticket_id).read_text()

    @activity.defn
    async def add_comment(self, ticket_id: str, comment: str) -> None:
        """Append ``comment`` to the ticket's comments."""
        path = self._ticket(ticket_id).with_suffix(".comments.jsonl")
        with path.open("a") as f:
            f.write(json.dumps(comment) + "\n")

    def _ticket(self, ticket_id: str) -> Path:
        """The ticket's file. Claude chose the id, so it is checked before it names a file."""
        if not _TICKET_ID.match(ticket_id):
            raise ApplicationError(
                f"{ticket_id!r} is not a ticket id like ENG-42", non_retryable=True
            )
        path = self._root / f"{ticket_id}.json"
        if not path.exists():
            raise ApplicationError(
                f"there is no ticket {ticket_id}", non_retryable=True
            )
        return path
