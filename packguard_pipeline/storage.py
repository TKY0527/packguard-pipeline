"""
In-memory lot store. Day 1 — replace with Redis/SQLite if needed for demo
robustness, but for the hackathon timeline this is enough.
"""

from threading import Lock
from typing import Optional

from .models import LotState


class LotStore:
    """Thread-safe in-memory dict of lot_id -> LotState."""

    def __init__(self) -> None:
        self._data: dict[str, LotState] = {}
        self._lock = Lock()

    def put(self, lot: LotState) -> None:
        with self._lock:
            self._data[lot.lot_id] = lot

    def get(self, lot_id: str) -> Optional[LotState]:
        with self._lock:
            return self._data.get(lot_id)

    def all_ids(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


# Module-level singleton — FastAPI app injects this via Depends.
_store = LotStore()


def get_store() -> LotStore:
    return _store
