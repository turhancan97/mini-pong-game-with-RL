"""Helper utilities shared by training and inference scripts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .processing import IMG_HISTORY


def stack_initial_frames(frame: np.ndarray) -> np.ndarray:
    """Create the initial state tensor by repeating the first frame."""
    stacked = np.stack([frame for _ in range(IMG_HISTORY)], axis=2)
    stacked = stacked.reshape(1, stacked.shape[0], stacked.shape[1], stacked.shape[2])
    return stacked


def append_frame(state: np.ndarray, new_frame: np.ndarray) -> np.ndarray:
    """Append a new frame to the state and drop the oldest one."""
    new_frame = new_frame.reshape(1, new_frame.shape[0], new_frame.shape[1], 1)
    return np.append(new_frame, state[:, :, :, : IMG_HISTORY - 1], axis=3)


@dataclass
class TrainingHistory:
    """Stores metrics collected during training."""

    steps: List[int] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)

    def record(self, step: int, score: float, loss: float) -> None:
        self.steps.append(step)
        self.scores.append(score)
        self.losses.append(loss)
