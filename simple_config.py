"""Simplified configs — only the knobs worth tuning.

Usage:
    from choi2025_follow_target.simple_config import SimpleSAC, SimplePPO, SimpleMMRKHS

    config = SimpleSAC(num_nodes=21, total_frames=5_000_000, learning_rate=1e-3)
    config = SimplePPO(num_nodes=21, total_frames=10_000_000, learning_rate=1e-4)
    config = SimpleMMRKHS(num_nodes=21, total_frames=5_000_000, learning_rate=3e-4)

Everything else inherits paper defaults from Choi & Tong (2025).
"""

from dataclasses import dataclass
from typing import Optional

from choi2025_follow_target.config import (
    Choi2025Config,
    Choi2025PPOConfig,
    Choi2025MMRKHSConfig,
)


def _wire_num_nodes(config, num_nodes: int) -> None:
    """Set num_nodes on the nested physics geometry."""
    config.env.physics.geometry.num_nodes = num_nodes


@dataclass
class SimpleSAC(Choi2025Config):
    """Simplified SAC config.

    Example::

        config = SimpleSAC(
            num_nodes=21,
            total_frames=5_000_000,
            learning_rate=1e-3,
            batch_size=2048,
            num_envs=32,
        )
    """

    num_nodes: int = 21
    learning_rate: float = 0.001
    total_frames: int = 20_000_000
    batch_size: int = 2048
    num_envs: int = 500
    max_wall_time: Optional[float] = None
    seed: int = 42

    def __post_init__(self):
        _wire_num_nodes(self, self.num_nodes)
        self.actor_lr = self.learning_rate
        self.critic_lr = self.learning_rate
        self.alpha_lr = self.learning_rate
        super().__post_init__()


@dataclass
class SimplePPO(Choi2025PPOConfig):
    """Simplified PPO config.

    Example::

        config = SimplePPO(
            num_nodes=21,
            total_frames=10_000_000,
            learning_rate=1e-4,
            num_epochs=10,
            mini_batch_size=1024,
            num_envs=32,
        )
    """

    num_nodes: int = 21
    learning_rate: float = 1e-4
    total_frames: int = 50_000_000
    num_epochs: int = 10
    mini_batch_size: int = 1024
    entropy_coef: float = 0.1
    num_envs: int = 500
    max_wall_time: Optional[float] = None
    seed: int = 42

    def __post_init__(self):
        _wire_num_nodes(self, self.num_nodes)
        super().__post_init__()


@dataclass
class SimpleMMRKHS(Choi2025MMRKHSConfig):
    """Simplified MM-RKHS config.

    Example::

        config = SimpleMMRKHS(
            num_nodes=21,
            total_frames=5_000_000,
            learning_rate=3e-4,
            num_epochs=10,
            mini_batch_size=1024,
            num_envs=32,
        )
    """

    num_nodes: int = 21
    learning_rate: float = 3e-4
    total_frames: int = 5_000_000
    num_epochs: int = 10
    mini_batch_size: int = 1024
    beta: float = 1.0
    eta: float = 1.0
    num_envs: int = 500
    max_wall_time: Optional[float] = None
    seed: int = 42

    def __post_init__(self):
        _wire_num_nodes(self, self.num_nodes)
        super().__post_init__()
