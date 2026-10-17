"""A Temporal External Storage driver that keeps payloads as files in a folder (tests only)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from temporalio.api.common.v1 import Payload
from temporalio.converter import (
    StorageDriver,
    StorageDriverClaim,
    StorageDriverRetrieveContext,
    StorageDriverStoreContext,
)


class FolderStorageDriver(StorageDriver):
    """Stores each payload in a file named by its SHA-256."""

    def __init__(self, folder: Path) -> None:
        """Create the driver.

        Args:
            folder: Where payloads go; created if missing.
        """
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)

    def name(self) -> str:
        """The driver's name in external storage references."""
        return "folder"

    async def store(
        self, context: StorageDriverStoreContext, payloads: Sequence[Payload]
    ) -> list[StorageDriverClaim]:
        """Write the payloads and return a claim for each."""
        del context
        claims: list[StorageDriverClaim] = []
        for payload in payloads:
            data = payload.SerializeToString()
            key = hashlib.sha256(data).hexdigest()
            (self.folder / key).write_bytes(data)
            claims.append(StorageDriverClaim(claim_data={"key": key}))
        return claims

    async def retrieve(
        self,
        context: StorageDriverRetrieveContext,
        claims: Sequence[StorageDriverClaim],
    ) -> list[Payload]:
        """Read the payloads back."""
        del context
        payloads: list[Payload] = []
        for claim in claims:
            payload = Payload()
            payload.ParseFromString(
                (self.folder / claim.claim_data["key"]).read_bytes()
            )
            payloads.append(payload)
        return payloads

    def stored_bytes(self) -> int:
        """Total bytes stored so far."""
        return sum(p.stat().st_size for p in self.folder.iterdir())
