"""Simple experience replay buffer."""
from __future__ import annotations

import random
from collections import deque
from typing import Deque, Iterable, Tuple

import numpy as np

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    """Fixed-size deque storing training transitions."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> Iterable[Transition]:
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)

    def to_list(self) -> Iterable[Transition]:
        return list(self._buffer)

    def load_from(self, data: Iterable[Transition]) -> None:
        self._buffer = deque(data, maxlen=self.capacity)
