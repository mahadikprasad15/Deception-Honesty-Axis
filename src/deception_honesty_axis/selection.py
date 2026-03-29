from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def select_items(items: Sequence[T], count: int, strategy: str) -> list[T]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if strategy != "first_n":
        raise ValueError(f"Unsupported selection strategy: {strategy}")
    return list(items[:count])
