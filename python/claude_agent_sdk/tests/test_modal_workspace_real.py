"""ModalVolumeWorkspaceStore against real Modal: round trip, concurrent writers, a lost blob,
and delete, on an ephemeral Volume and Dict that vanish with the test. Confirms what the
offline fake assumes (paths, listing a missing directory, batch atomicity, Dict.put's
``skip_if_exists``). Requires ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from temporalio.claude_agent_sdk import ModalVolumeWorkspaceStore
from temporalio.claude_agent_sdk._workspace import WorkspaceIntegrityError

from .helpers.fixtures import POSIX_ONLY
from .test_claude_workspace import SID, make_workspace, tree

modal = pytest.importorskip("modal", reason="the modal extra is not installed")

pytestmark = [
    POSIX_ONLY,
    pytest.mark.skipif(
        not (os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")),
        reason="real Modal: set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET",
    ),
]


@pytest.mark.timeout(300)
def test_real_modal_round_trip_concurrency_and_delete(tmp_path: Path) -> None:
    with modal.Volume.ephemeral() as volume, modal.Dict.ephemeral() as claims:
        asyncio.run(_scenario(tmp_path, volume, claims))


async def _scenario(tmp_path: Path, volume: Any, claims: Any) -> None:
    store = ModalVolumeWorkspaceStore(volume, claims)
    assert await store.revisions(SID) == []  # listing a directory that does not exist
    assert await store.latest(SID) is None

    source = make_workspace(tmp_path / "a" / SID)
    assert (await store.persist(SID, source)).revision == 1
    assert (await store.persist(SID, source)).revision == 1  # unchanged
    target = tmp_path / "b" / SID
    await ModalVolumeWorkspaceStore(volume, claims).restore(SID, 1, target)
    assert tree(target) == tree(source)

    sources = [
        make_workspace(tmp_path / f"w{i}", marker=f"writer {i}") for i in range(4)
    ]
    stores = [ModalVolumeWorkspaceStore(volume, claims) for _ in range(2)]
    snapshots = await asyncio.gather(
        *(stores[i % 2].persist(SID, s) for i, s in enumerate(sources))
    )
    assert sorted(s.revision for s in snapshots) == [2, 3, 4, 5]
    for snapshot, s in zip(snapshots, sources, strict=True):
        restored = tmp_path / f"r{snapshot.revision}"
        await store.restore(SID, snapshot.revision, restored)
        assert tree(restored) == tree(s)

    first = await store.get(SID, 1)
    await volume.remove_file.aio(f"/{SID}/blobs/{first.sha256}.tar.gz")
    with pytest.raises(WorkspaceIntegrityError, match="missing its archive"):
        await store.restore(SID, 1, target)
    assert tree(target) == tree(source)

    await store.delete(SID)
    assert await store.revisions(SID) == []
    with pytest.raises(KeyError):
        await store.get(SID, 1)
    assert (await store.persist(SID, source)).revision == 1
