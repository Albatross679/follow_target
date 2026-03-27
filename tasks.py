"""Task logic for soft manipulator tasks (Choi & Tong, 2025).

Provides target generation and obstacle management.
"""

import numpy as np

from choi2025_follow_target.config import (
    ObstacleConfig,
    TargetConfig,
    TaskType,
)


class TargetGenerator:
    """Generates and updates targets within the manipulator workspace."""

    def __init__(self, config: TargetConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

        self.position = np.zeros(3)
        self.orientation = np.array([1.0, 0.0, 0.0])
        self._velocity = np.zeros(3)

    def sample(self, task: TaskType, speed_override: float | None = None) -> None:
        """Sample a new target appropriate for the given task."""
        r = self.rng.uniform(self.config.min_radius, self.config.max_radius)
        theta = self.rng.uniform(0, 2 * np.pi)
        phi = self.rng.uniform(0, np.pi)

        self.position = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi),
        ])

        if task == TaskType.INVERSE_KINEMATICS:
            orient = self.rng.standard_normal(3)
            orient /= np.linalg.norm(orient) + 1e-8
            self.orientation = orient

        if task == TaskType.FOLLOW_TARGET:
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


class ObstacleManager:
    """Manages obstacles for obstacle avoidance tasks."""

    def __init__(self, config: ObstacleConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

        self.positions: np.ndarray = np.zeros((0, 3))
        self.radii: np.ndarray = np.zeros(0)

    def setup(self, task: TaskType) -> None:
        """Generate obstacle layout for the given task."""
        if task == TaskType.TIGHT_OBSTACLES:
            self._setup_tight_obstacles()
        elif task == TaskType.RANDOM_OBSTACLES:
            self._setup_random_obstacles()
        else:
            self.positions = np.zeros((0, 3))
            self.radii = np.zeros(0)

    def _setup_tight_obstacles(self) -> None:
        gap_half = self.config.gap_width / 2
        r = self.config.obstacle_radius
        dist = (self.config.min_distance + self.config.max_distance) / 2

        self.positions = np.array([
            [dist, gap_half + r, 0.0],
            [dist, -(gap_half + r), 0.0],
        ])
        self.radii = np.array([r, r])

    def _setup_random_obstacles(self) -> None:
        n = self.config.num_obstacles
        positions = []
        radii = []

        for _ in range(n):
            r = self.rng.uniform(self.config.min_distance, self.config.max_distance)
            theta = self.rng.uniform(0, 2 * np.pi)
            phi = self.rng.uniform(0.3, np.pi - 0.3)

            pos = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi),
            ])
            positions.append(pos)
            radii.append(self.config.obstacle_radius)

        self.positions = np.array(positions) if positions else np.zeros((0, 3))
        self.radii = np.array(radii) if radii else np.zeros(0)

    def compute_penetrations(self, rod_positions: np.ndarray) -> np.ndarray:
        """Compute penetration depths between rod nodes and obstacles."""
        if len(self.positions) == 0:
            return np.zeros(rod_positions.shape[0])

        penetrations = np.zeros(rod_positions.shape[0])

        for obs_pos, obs_r in zip(self.positions, self.radii):
            diffs = rod_positions - obs_pos[np.newaxis, :]
            dists = np.linalg.norm(diffs, axis=1)
            pen = np.maximum(0.0, obs_r - dists)
            penetrations += pen

        return penetrations
