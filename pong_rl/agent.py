"""Deep Q-learning agent for Pong."""
from __future__ import annotations

import pickle
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten
from tensorflow.keras.models import clone_model

from .replay_buffer import ReplayBuffer, Transition


@dataclass
class AgentConfig:
    n_actions: int = 3
    img_height: int = 40
    img_width: int = 40
    img_history: int = 4
    observe_steps: int = 2000
    gamma: float = 0.975
    batch_size: int = 64
    replay_capacity: int = 2000
    initial_epsilon: float = 1.0
    target_update_tau: float = 0.01
    target_update_interval: int = 1000
    learning_rate: float = 0.00025
    min_replay_size: int = 1000


class DeepQAgent:
    """Deep Q-learning agent with experience replay and a target network."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

        tf.compat.v1.disable_eager_execution()

        self.online_network = self._build_network()
        self.target_network = clone_model(self.online_network)
        self.target_network.set_weights(self.online_network.get_weights())

        self.replay_buffer = ReplayBuffer(capacity=config.replay_capacity)

        self.steps = 0
        self.train_updates = 0
        self.epsilon = config.initial_epsilon

    def _build_network(self) -> Sequential:
        cfg = self.config

        model = Sequential()
        model.add(
            Conv2D(
                32,
                kernel_size=4,
                strides=(2, 2),
                input_shape=(cfg.img_height, cfg.img_width, cfg.img_history),
                padding="same",
            )
        )
        model.add(Activation("relu"))
        model.add(Conv2D(64, kernel_size=4, strides=(2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64, kernel_size=3, strides=(1, 1), padding="same"))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dense(units=cfg.n_actions, activation="linear"))

        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
        model.compile(loss="mse", optimizer=optimizer)
        return model

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon or self.steps < self.config.observe_steps:
            return random.randint(0, self.config.n_actions - 1)

        q_values = self.online_network.predict(state, verbose=0)
        return int(np.argmax(q_values[0]))

    def select_greedy_action(self, state: np.ndarray) -> int:
        q_values = self.online_network.predict(state, verbose=0)
        return int(np.argmax(q_values[0]))

    def store_transition(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> None:
        self.replay_buffer.push((state, action, reward, next_state, done))
        self.steps += 1
        self._update_epsilon()

    def _update_epsilon(self) -> None:
        if self.steps <= self.config.observe_steps:
            self.epsilon = 1.0
            return

        if self.steps > 70000:
            self.epsilon = 0.05
        elif self.steps > 45000:
            self.epsilon = 0.1
        elif self.steps > 30000:
            self.epsilon = 0.15
        elif self.steps > 14000:
            self.epsilon = 0.25
        elif self.steps > 7000:
            self.epsilon = 0.5
        else:
            self.epsilon = 0.75

    def can_train(self) -> bool:
        return (
            len(self.replay_buffer) >= max(self.config.batch_size, self.config.min_replay_size)
            and self.steps > self.config.observe_steps
        )

    def train_step(self) -> float:
        if not self.can_train():
            return 0.0

        transitions: Iterable[Transition] = self.replay_buffer.sample(self.config.batch_size)
        batch = list(transitions)

        states = np.concatenate([t[0] for t in batch], axis=0)
        next_states = np.concatenate([t[3] for t in batch], axis=0)

        targets = self.online_network.predict(states, verbose=0)
        next_q_values = self.target_network.predict(next_states, verbose=0)

        for idx, (_, action, reward, _, done) in enumerate(batch):
            if done:
                targets[idx, action] = reward
            else:
                targets[idx, action] = reward + self.config.gamma * np.max(next_q_values[idx])

        history = self.online_network.fit(
            states, targets, batch_size=self.config.batch_size, epochs=1, verbose=0
        )

        self.train_updates += 1
        if self.train_updates % self.config.target_update_interval == 0:
            self._soft_update_target()

        return float(history.history["loss"][0])

    def _soft_update_target(self) -> None:
        tau = self.config.target_update_tau
        online_weights = self.online_network.get_weights()
        target_weights = self.target_network.get_weights()
        updated_weights = []
        for target_w, online_w in zip(target_weights, online_weights):
            updated_weights.append(tau * online_w + (1.0 - tau) * target_w)
        self.target_network.set_weights(updated_weights)

    def save_model(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        self.online_network.save(path)

    def save_checkpoint(self, checkpoint_dir: Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_path = checkpoint_dir / "model"
        if model_path.exists():
            if model_path.is_dir():
                shutil.rmtree(model_path)
            else:
                model_path.unlink()
        self.online_network.save(model_path)

        state = {
            "steps": self.steps,
            "train_updates": self.train_updates,
            "epsilon": self.epsilon,
            "buffer": self.replay_buffer.to_list(),
        }

        with open(checkpoint_dir / "agent_state.pkl", "wb") as fh:
            pickle.dump(state, fh)

    def load_checkpoint(self, checkpoint_dir: Path) -> bool:
        checkpoint_dir = Path(checkpoint_dir)
        model_path = checkpoint_dir / "model"
        state_path = checkpoint_dir / "agent_state.pkl"

        if not model_path.exists() or not state_path.exists():
            return False

        self.online_network = tf.keras.models.load_model(model_path)
        self.target_network = clone_model(self.online_network)
        self.target_network.set_weights(self.online_network.get_weights())

        with open(state_path, "rb") as fh:
            state = pickle.load(fh)

        self.steps = state.get("steps", 0)
        self.train_updates = state.get("train_updates", 0)
        self.epsilon = state.get("epsilon", self.config.initial_epsilon)
        buffer_data = state.get("buffer", [])
        self.replay_buffer.load_from(buffer_data)
        return True

    def load_model(self, path: Path) -> None:
        self.online_network = tf.keras.models.load_model(path)
        self.target_network = clone_model(self.online_network)
        self.target_network.set_weights(self.online_network.get_weights())

    def get_networks(self) -> Tuple[Sequential, Sequential]:
        return self.online_network, self.target_network
