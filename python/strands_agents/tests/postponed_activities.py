"""A callable-class activity in a module with postponed annotations."""

from __future__ import annotations

from dataclasses import dataclass

from temporalio import activity


@dataclass
class Parcel:
    parcel_id: str


@activity.defn(name="load_parcels")
class LoadParcels:
    async def __call__(self, parcels: list[Parcel]) -> str:
        return ", ".join(parcel.parcel_id for parcel in parcels)
