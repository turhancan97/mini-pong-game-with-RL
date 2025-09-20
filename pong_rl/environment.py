"""Gameplay environment for the Pong agent."""
from __future__ import annotations

import os
import random
from typing import Tuple

import pygame
import numpy as np

FPS = 20
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 420
GAME_HEIGHT = 400
PADDLE_WIDTH = 15
PADDLE_HEIGHT = 60
PADDLE_BUFFER = 15
BALL_WIDTH = 20
BALL_HEIGHT = 20
PADDLE_SPEED = 3
BALL_X_SPEED = 2
BALL_Y_SPEED = 2
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class PongEnvironment:
    """Wraps the classic Pong game implemented with pygame."""

    def __init__(self, render: bool = True) -> None:
        self.render_enabled = render

        if not render:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        pygame.init()
        pygame.display.set_caption("Pong Game")

        if render:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        else:
            self.screen = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))

        self.clock = pygame.time.Clock()

        self.paddle1_y = GAME_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.paddle2_y = GAME_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.ball_x = WINDOW_WIDTH / 2
        self.ball_y = random.randint(0, 9) * (WINDOW_HEIGHT - BALL_HEIGHT) / 9
        self.ball_x_dir = random.choice([-1, 1])
        self.ball_y_dir = random.choice([-1, 1])

        self.score_smoothed = 0.0

        self._initial_render()

    def reset(self) -> Tuple[float, np.ndarray]:
        """Reset the game to a fresh state."""
        self.paddle1_y = GAME_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.paddle2_y = GAME_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.ball_x = WINDOW_WIDTH / 2
        self.ball_y = random.randint(0, 9) * (WINDOW_HEIGHT - BALL_HEIGHT) / 9
        self.ball_x_dir = random.choice([-1, 1])
        self.ball_y_dir = random.choice([-1, 1])
        self.score_smoothed = 0.0

        self._initial_render()
        frame = self._capture_frame()
        return 0.0, frame

    def step(self, action: int) -> Tuple[float, np.ndarray, float, bool]:
        """Advance the simulation by one action step."""
        delta_frame_time = self.clock.tick(FPS)

        pygame.event.pump()
        self.screen.fill(BLACK)

        self.paddle1_y = _update_paddle("left", action, self.paddle1_y, self.ball_y)
        _draw_paddle(self.screen, "left", self.paddle1_y)

        self.paddle2_y = _update_paddle("right", action, self.paddle2_y, self.ball_y)
        _draw_paddle(self.screen, "right", self.paddle2_y)

        score, self.ball_x, self.ball_y, self.ball_x_dir, self.ball_y_dir = _update_ball(
            self.paddle1_y,
            self.paddle2_y,
            self.ball_x,
            self.ball_y,
            self.ball_x_dir,
            self.ball_y_dir,
            delta_frame_time,
        )

        _draw_ball(self.screen, self.ball_x, self.ball_y)

        if score > 0.5 or score < -0.5:
            self.score_smoothed = self.score_smoothed * 0.9 + 0.1 * score

        if self.render_enabled:
            pygame.display.flip()

        frame = self._capture_frame()
        done = False
        return score, frame, self.score_smoothed, done

    def close(self) -> None:
        pygame.quit()

    def _initial_render(self) -> None:
        pygame.event.pump()
        self.screen.fill(BLACK)
        _draw_paddle(self.screen, "left", self.paddle1_y)
        _draw_paddle(self.screen, "right", self.paddle2_y)
        _draw_ball(self.screen, self.ball_x, self.ball_y)
        if self.render_enabled:
            pygame.display.flip()

    def _capture_frame(self) -> np.ndarray:
        if self.render_enabled:
            surface = pygame.display.get_surface()
            assert surface is not None
            return pygame.surfarray.array3d(surface)
        # For headless mode we need to copy from the off-screen surface.
        return pygame.surfarray.array3d(self.screen)


def _draw_paddle(screen: pygame.Surface, side: str, paddle_y: float) -> None:
    if side == "left":
        paddle_rect = pygame.Rect(PADDLE_BUFFER, paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT)
    else:
        paddle_rect = pygame.Rect(
            WINDOW_WIDTH - PADDLE_BUFFER - PADDLE_WIDTH,
            paddle_y,
            PADDLE_WIDTH,
            PADDLE_HEIGHT,
        )
    pygame.draw.rect(screen, WHITE, paddle_rect)


def _draw_ball(screen: pygame.Surface, ball_x: float, ball_y: float) -> None:
    ball = pygame.Rect(ball_x, ball_y, BALL_WIDTH, BALL_HEIGHT)
    pygame.draw.rect(screen, WHITE, ball)


def _update_paddle(side: str, action: int, paddle_y: float, ball_y: float) -> float:
    dft = 7.5

    if side == "left":
        if action == 1:
            paddle_y -= PADDLE_SPEED * dft
        elif action == 2:
            paddle_y += PADDLE_SPEED * dft

        paddle_y = max(0, min(GAME_HEIGHT - PADDLE_HEIGHT, paddle_y))
    else:
        center_diff = (ball_y + BALL_HEIGHT / 2) - (paddle_y + PADDLE_HEIGHT / 2)
        if center_diff > 0:
            paddle_y += PADDLE_SPEED * dft
        elif center_diff < 0:
            paddle_y -= PADDLE_SPEED * dft

        paddle_y = max(0, min(GAME_HEIGHT - PADDLE_HEIGHT, paddle_y))

    return paddle_y


def _update_ball(
    paddle1_y: float,
    paddle2_y: float,
    ball_x: float,
    ball_y: float,
    ball_x_dir: int,
    ball_y_dir: int,
    delta_frame_time: float,
) -> Tuple[float, float, float, int, int]:
    _ = delta_frame_time
    dft = 7.5

    new_x = ball_x + ball_x_dir * BALL_X_SPEED * dft
    new_y = ball_y + ball_y_dir * BALL_Y_SPEED * dft

    score = -0.05

    # Vertical wall collisions
    if new_y <= 0:
        new_y = 0
        ball_y_dir = 1
    elif new_y >= GAME_HEIGHT - BALL_HEIGHT:
        new_y = GAME_HEIGHT - BALL_HEIGHT
        ball_y_dir = -1

    # Left paddle (agent) collision or miss
    left_paddle_edge = PADDLE_BUFFER + PADDLE_WIDTH
    if ball_x_dir == -1 and new_x <= left_paddle_edge:
        paddle_top = paddle1_y
        paddle_bottom = paddle1_y + PADDLE_HEIGHT
        ball_top = new_y
        ball_bottom = new_y + BALL_HEIGHT

        if ball_bottom >= paddle_top and ball_top <= paddle_bottom:
            new_x = left_paddle_edge
            ball_x_dir = 1
            score = 10
        else:
            new_x = PADDLE_BUFFER
            ball_x_dir = 1
            return -10, new_x, new_y, ball_x_dir, ball_y_dir

    # Right paddle (opponent) collision or wall
    right_paddle_edge = WINDOW_WIDTH - PADDLE_BUFFER - PADDLE_WIDTH
    if ball_x_dir == 1 and new_x + BALL_WIDTH >= right_paddle_edge:
        paddle_top = paddle2_y
        paddle_bottom = paddle2_y + PADDLE_HEIGHT
        ball_top = new_y
        ball_bottom = new_y + BALL_HEIGHT

        if ball_bottom >= paddle_top and ball_top <= paddle_bottom:
            new_x = right_paddle_edge - BALL_WIDTH
            ball_x_dir = -1
        else:
            new_x = WINDOW_WIDTH - PADDLE_BUFFER - BALL_WIDTH
            ball_x_dir = -1
            return score, new_x, new_y, ball_x_dir, ball_y_dir

    # Horizontal wall out-of-bounds (failsafe)
    if new_x <= 0:
        new_x = 0
        ball_x_dir = 1
    elif new_x >= WINDOW_WIDTH - BALL_WIDTH:
        new_x = WINDOW_WIDTH - BALL_WIDTH
        ball_x_dir = -1

    return score, new_x, new_y, ball_x_dir, ball_y_dir
