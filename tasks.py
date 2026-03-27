"""Task logic for soft manipulator tasks (Choi & Tong, 2025).

Provides target generation and obstacle management.
"""

import numpy as np

from .config import TargetConfig


class TargetGenerator:
    """Generates and updates targets within the manipulator workspace."""

    def __init__(self, config: TargetConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

        self.position = np.zeros(3)
        self._velocity = np.zeros(3)

    def sample(self, speed_override: float | None = None) -> None:
        """Sample a new target for follow_target task."""
        r = self.rng.uniform(self.config.min_radius, self.config.max_radius)
        theta = self.rng.uniform(0, 2 * np.pi)
        phi = self.rng.uniform(0, np.pi)

        self.position = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi),
        ])

        speed = speed_override if speed_override is not None else self.config.target_speed
        vel_dir = self.rng.standard_normal(3)
        vel_dir /= np.linalg.norm(vel_dir) + 1e-8
        self._velocity = vel_dir * speed

    def step(self, dt: float) -> None:
        """Move target for follow_target task (bounces off workspace boundary)."""
        new_pos = self.position + self._velocity * dt
        r = np.linalg.norm(new_pos)

        if r > self.config.max_radius:
            normal = new_pos / (r + 1e-8)
            self._velocity -= 2 * np.dot(self._velocity, normal) * normal
            new_pos = self.position + self._velocity * dt

        self.position = new_pos
