"""Configuration dataclasses for soft manipulator control (Choi & Tong, 2025).

Self-contained — all parent config classes are inlined so this package has
no dependency on ``src/``.

Hierarchy:
    GeometryConfig          → rod geometry
    PhysicsConfig           → base physics (dt, gravity, density)
    Choi2025PhysicsConfig   → 3D clamped manipulator overrides
    DeltaCurvatureControlConfig → control point interpolation
    TargetConfig            → target sampling ranges
    ObstacleConfig          → obstacle layout
    Choi2025EnvConfig       → composes physics + control + target + obstacles
    NetworkConfig           → actor/critic architecture
    SACConfig               → SAC hyperparameters
    Choi2025Config          → top-level project config
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import json
import torch
from dataclasses import asdict
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    """Manipulation task types from Choi & Tong (2025)."""

    FOLLOW_TARGET = "follow_target"
    INVERSE_KINEMATICS = "inverse_kinematics"
    TIGHT_OBSTACLES = "tight_obstacles"
    RANDOM_OBSTACLES = "random_obstacles"


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def resolve_device(device: str = "auto") -> str:
    """Resolve a device string to an actual torch device name."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# ---------------------------------------------------------------------------
# Geometry config
# ---------------------------------------------------------------------------


@dataclass
class GeometryConfig:
    """Snake body geometry."""

    snake_length: float = 1.0
    snake_radius: float = 0.001
    num_segments: int = 20

    @property
    def num_nodes(self) -> int:
        return self.num_segments + 1

    @property
    def segment_length(self) -> float:
        return self.snake_length / self.num_segments


# ---------------------------------------------------------------------------
# Physics config (simplified for this package)
# ---------------------------------------------------------------------------


@dataclass
class PhysicsConfig:
    """Base physics config."""

    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    dt: float = 5e-2
    density: float = 1200.0
    enable_gravity: bool = True
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)

    @property
    def snake_length(self) -> float:
        return self.geometry.snake_length

    @property
    def snake_radius(self) -> float:
        return self.geometry.snake_radius

    @property
    def num_segments(self) -> int:
        return self.geometry.num_segments


@dataclass
class Choi2025PhysicsConfig(PhysicsConfig):
    """DisMech physics for clamped soft manipulator."""

    clamp_first_node: bool = True
    two_d_sim: bool = False

    # Contact parameters
    contact_stiffness: float = 1e6
    contact_delta: float = 0.005
    max_newton_iter_contact: int = 5
    max_newton_iter_noncontact: int = 2

    # Override rod defaults
    geometry: GeometryConfig = field(
        default_factory=lambda: GeometryConfig(
            snake_length=1.0,
            snake_radius=0.05,
            num_segments=20,
        )
    )
    dt: float = 0.05
    youngs_modulus: float = 10e6
    poisson_ratio: float = 0.5
    density: float = 1000.0

    enable_gravity: bool = True
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)

    # Newton solver convergence
    tol: float = 1e-4
    ftol: float = 1e-4
    dtol: float = 1e-2


# ---------------------------------------------------------------------------
# Control config
# ---------------------------------------------------------------------------


@dataclass
class DeltaCurvatureControlConfig:
    """Configuration for delta curvature control with Voronoi smoothing."""

    num_control_points: int = 5
    max_delta_curvature: float = 1.0
    voronoi_smoothing: bool = True


# ---------------------------------------------------------------------------
# Target and obstacle configs
# ---------------------------------------------------------------------------


@dataclass
class CurriculumConfig:
    """Curriculum learning: ramp target speed over warmup episodes."""

    enabled: bool = False
    warmup_episodes: int = 200
    initial_speed_frac: float = 0.2


@dataclass
class TargetConfig:
    """Target sampling configuration."""

    min_radius: float = 0.3
    max_radius: float = 0.9
    target_speed: float = 0.05
    match_orientation: bool = False
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)


@dataclass
class ObstacleConfig:
    """Obstacle configuration for obstacle avoidance tasks."""

    num_obstacles: int = 3
    obstacle_radius: float = 0.05
    min_distance: float = 0.2
    max_distance: float = 0.8
    gap_width: float = 0.15
    contact_penalty: float = 10.0


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------


@dataclass
class Choi2025EnvConfig:
    """Environment configuration for soft manipulator tasks."""

    physics: Choi2025PhysicsConfig = field(default_factory=Choi2025PhysicsConfig)
    control: DeltaCurvatureControlConfig = field(default_factory=DeltaCurvatureControlConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    obstacles: ObstacleConfig = field(default_factory=ObstacleConfig)

    task: TaskType = TaskType.FOLLOW_TARGET
    max_episode_steps: int = 200
    control_period: int = 2

    heading_weight: float = 0.0
    pbrs_gamma: float = 0.0
    pbrs_only: bool = False
    improvement_weight: float = 0.0

    device: str = "auto"


# ---------------------------------------------------------------------------
# Network config
# ---------------------------------------------------------------------------


@dataclass
class ActorConfig:
    """Configuration for actor (policy) network."""

    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128])
    activation: str = "tanh"
    distribution: str = "tanh_normal"
    min_std: float = 0.1
    max_std: float = 1.0
    init_std: float = 0.5
    ortho_init: bool = True
    init_gain: float = 0.01
    use_layer_norm: bool = False
    use_spectral_norm: bool = False


@dataclass
class CriticConfig:
    """Configuration for critic (value) network."""

    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128])
    activation: str = "tanh"
    num_outputs: int = 1
    ortho_init: bool = True
    init_gain: float = 1.0
    use_layer_norm: bool = False
    use_spectral_norm: bool = False


@dataclass
class NetworkConfig:
    """Combined network configuration."""

    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    device: str = "cpu"
    dtype: str = "float32"


@dataclass
class Choi2025NetworkConfig(NetworkConfig):
    """Network config: 3x256 ReLU MLP (Table A.1)."""

    actor: ActorConfig = field(
        default_factory=lambda: ActorConfig(
            hidden_dims=[256, 256, 256],
            activation="relu",
            ortho_init=True,
            init_gain=0.01,
            min_std=0.1,
            max_std=1.0,
            init_std=0.5,
        )
    )
    critic: CriticConfig = field(
        default_factory=lambda: CriticConfig(
            hidden_dims=[256, 256, 256],
            activation="relu",
            ortho_init=True,
            init_gain=1.0,
        )
    )


@dataclass
class Choi2025PPONetworkConfig(NetworkConfig):
    """Network config for PPO: 4x512 ReLU MLP."""

    actor: ActorConfig = field(
        default_factory=lambda: ActorConfig(
            hidden_dims=[512, 512, 512, 512],
            activation="relu",
            ortho_init=True,
            init_gain=0.01,
            min_std=0.2,
            max_std=1.0,
            init_std=0.5,
        )
    )
    critic: CriticConfig = field(
        default_factory=lambda: CriticConfig(
            hidden_dims=[512, 512, 512, 512],
            activation="relu",
            ortho_init=True,
            init_gain=1.0,
        )
    )


# ---------------------------------------------------------------------------
# Logging configs
# ---------------------------------------------------------------------------


@dataclass
class WandB:
    """Weights & Biases logging configuration."""

    enabled: bool = True
    project: str = "choi2025-follow-target"
    entity: str = ""
    group: str = ""
    tags: list = field(default_factory=list)


@dataclass
class Output:
    """Run output directory configuration."""

    base_dir: str = "output"
    save_config: bool = True


@dataclass
class Console:
    """Console logging configuration."""

    enabled: bool = True
    filename: str = "console.log"
    tee_to_console: bool = True
    line_timestamps: bool = False
    timestamp_format: str = "%H:%M:%S"


# ---------------------------------------------------------------------------
# Training config (SAC)
# ---------------------------------------------------------------------------


@dataclass
class SACConfig:
    """SAC training configuration."""

    name: str = "experiment"
    seed: int = 42
    device: str = "auto"

    # Training duration
    total_frames: int = 1_000_000
    max_wall_time: Optional[float] = None

    # SAC hyperparameters
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: Optional[float] = None
    alpha_lr: float = 3e-4

    # Replay buffer
    buffer_size: int = 1_000_000
    batch_size: int = 256

    # Training
    warmup_steps: int = 10000
    update_frequency: int = 1
    num_updates: int = 1

    # Target network
    soft_update_period: int = 1

    # Critic
    num_critics: int = 2
    critic_lr: float = 3e-4

    # Actor
    actor_lr: float = 3e-4
    actor_update_frequency: int = 2
    actor_max_grad_norm: Optional[float] = None
    max_grad_norm: float = 0.5

    # Mixed precision
    use_amp: bool = False

    # Logging
    log_interval: int = 1
    save_interval: int = 100

    # Parallelism
    num_envs: int = 1


# ---------------------------------------------------------------------------
# PPO training config
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    """PPO training configuration."""

    name: str = "experiment"
    seed: int = 42
    device: str = "auto"

    # Training duration
    total_frames: int = 1_000_000
    max_wall_time: Optional[float] = None
    frames_per_batch: int = 4096

    # Optimization
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    weight_decay: float = 0.0
    num_epochs: int = 10
    mini_batch_size: int = 256

    # PPO hyperparameters
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantage: bool = True

    # Value function clipping
    clip_value_loss: bool = True
    value_clip_epsilon: float = 0.2

    # Policy constraints
    target_kl: Optional[float] = 0.01

    # Learning rate schedule
    lr_schedule: str = "linear"
    lr_end: float = 1e-5

    # Early stopping
    patience_batches: int = 200

    # Mixed precision
    use_amp: bool = False

    # Logging
    log_interval: int = 1
    save_interval: int = 100

    # Parallelism
    num_envs: int = 1


# ---------------------------------------------------------------------------
# MM-RKHS training config
# ---------------------------------------------------------------------------


@dataclass
class MMRKHSConfig:
    """MM-RKHS configuration (Gupta & Mahajan, 2026, arXiv:2603.17875)."""

    name: str = "experiment"
    seed: int = 42
    device: str = "auto"

    # Training duration
    total_frames: int = 1_000_000
    max_wall_time: Optional[float] = None
    frames_per_batch: int = 4096

    # Optimization
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    weight_decay: float = 0.0
    num_epochs: int = 10
    mini_batch_size: int = 256
    gamma: float = 0.99

    # MM-RKHS specific
    beta: float = 1.0
    eta: float = 1.0
    mmd_kernel: str = "rbf"
    mmd_bandwidth: float = 1.0
    mmd_num_samples: int = 16

    # Notebook mechanics (opt-in)
    eta_schedule: bool = False
    eta_exponent: float = 2.0
    beta_schedule: bool = False
    inner_mm_iterations: int = 1
    exponent_clip: float = 20.0
    kernel_correction: bool = False
    kernel_correction_weight: float = 1.0

    # GAE
    gae_lambda: float = 0.95
    normalize_advantage: bool = True

    # Critic
    value_coef: float = 0.5

    # Learning rate schedule
    lr_schedule: str = "linear"
    lr_end: float = 1e-5

    # Early stopping
    patience_batches: int = 200

    # Mixed precision
    use_amp: bool = False

    # Logging
    log_interval: int = 1
    save_interval: int = 100

    # Parallelism
    num_envs: int = 1


# ---------------------------------------------------------------------------
# Top-level configs
# ---------------------------------------------------------------------------


@dataclass
class Choi2025Config(SACConfig):
    """Top-level config for soft manipulator SAC training.

    Overrides SAC defaults to match Choi & Tong (2025):
    - lr=0.001, batch=2048, buffer=2M, UTD=4
    """

    name: str = "choi2025"
    experiment_name: str = "choi2025"

    # SAC hyperparameters from paper (Table A.1)
    total_frames: int = 20_000_000
    actor_lr: float = 0.001
    critic_lr: float = 0.001
    alpha_lr: float = 0.001
    batch_size: int = 2048
    buffer_size: int = 2_000_000
    num_updates: int = 4
    warmup_steps: int = 1000

    auto_alpha: bool = False
    alpha: float = 0.0

    soft_update_period: int = 8
    use_amp: bool = False

    max_grad_norm: float = None
    actor_max_grad_norm: float = 1.0
    actor_update_frequency: int = 1

    env: Choi2025EnvConfig = field(default_factory=Choi2025EnvConfig)
    network: Choi2025NetworkConfig = field(default_factory=Choi2025NetworkConfig)
    wandb: WandB = field(default_factory=lambda: WandB(project="choi2025-follow-target"))
    output: Output = field(default_factory=Output)
    console: Console = field(default_factory=Console)

    num_envs: int = 500

    def __post_init__(self):
        """Set name and experiment_name from task."""
        task = self.env.task.value if isinstance(self.env.task, TaskType) else self.env.task
        self.name = f"fixed_{task}_sac_lr1e3_{self.num_envs}envs"
        self.experiment_name = self.name


@dataclass
class Choi2025PPOConfig(PPOConfig):
    """Top-level config for soft manipulator PPO training."""

    name: str = "choi2025_ppo"
    experiment_name: str = "choi2025_ppo"

    total_frames: int = 50_000_000
    learning_rate: float = 1e-4
    clip_epsilon: float = 0.2
    num_epochs: int = 10
    mini_batch_size: int = 1024
    frames_per_batch: int = 8192
    gae_lambda: float = 0.95
    entropy_coef: float = 0.1
    value_coef: float = 0.5
    normalize_advantage: bool = True
    patience_batches: int = 0

    env: Choi2025EnvConfig = field(default_factory=Choi2025EnvConfig)
    network: Choi2025PPONetworkConfig = field(default_factory=Choi2025PPONetworkConfig)
    wandb: WandB = field(default_factory=lambda: WandB(project="choi2025-follow-target"))
    output: Output = field(default_factory=Output)
    console: Console = field(default_factory=Console)

    num_envs: int = 500

    def __post_init__(self):
        """Set name and experiment_name from task."""
        task = self.env.task.value if isinstance(self.env.task, TaskType) else self.env.task
        self.name = f"fixed_{task}_ppo_lr1e4_{self.num_envs}envs"
        self.experiment_name = self.name


@dataclass
class Choi2025MMRKHSConfig(MMRKHSConfig):
    """Top-level config for soft manipulator MM-RKHS training."""

    name: str = "choi2025_mmrkhs"
    experiment_name: str = "choi2025_mmrkhs"

    total_frames: int = 5_000_000
    learning_rate: float = 3e-4
    num_epochs: int = 10
    mini_batch_size: int = 1024
    frames_per_batch: int = 8192
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    patience_batches: int = 0

    # MM-RKHS specific
    beta: float = 1.0
    eta: float = 1.0
    mmd_bandwidth: float = 1.0
    mmd_num_samples: int = 16
    value_coef: float = 0.5

    # Notebook mechanics (enabled by default)
    eta_schedule: bool = True
    eta_exponent: float = 2.0
    beta_schedule: bool = True
    inner_mm_iterations: int = 3
    exponent_clip: float = 2.0
    kernel_correction: bool = True
    kernel_correction_weight: float = 1.0

    env: Choi2025EnvConfig = field(default_factory=Choi2025EnvConfig)
    network: Choi2025NetworkConfig = field(default_factory=Choi2025NetworkConfig)
    wandb: WandB = field(default_factory=lambda: WandB(project="choi2025-follow-target"))
    output: Output = field(default_factory=Output)
    console: Console = field(default_factory=Console)

    num_envs: int = 500

    def __post_init__(self):
        """Set name and experiment_name from task."""
        task = self.env.task.value if isinstance(self.env.task, TaskType) else self.env.task
        self.name = f"fixed_{task}_mmrkhs_lr3e4_{self.num_envs}envs"
        self.experiment_name = self.name


# ---------------------------------------------------------------------------
# Run directory setup
# ---------------------------------------------------------------------------


def setup_run_dir(config, *, timestamp=None, base_dir=None):
    """Create a timestamped run directory and save a config snapshot."""
    if base_dir is None:
        if hasattr(config, "output") and hasattr(config.output, "base_dir"):
            base_dir = config.output.base_dir
        else:
            base_dir = "output"

    name = getattr(config, "name", "run")

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = Path(base_dir) / f"{name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    save_config = True
    if hasattr(config, "output") and hasattr(config.output, "save_config"):
        save_config = config.output.save_config

    if save_config:
        config_path = run_dir / "config.json"
        config_path.write_text(
            json.dumps(asdict(config), indent=2, default=str)
        )

    return run_dir


# ---------------------------------------------------------------------------
# Config serialization helpers
# ---------------------------------------------------------------------------


def save_config(cfg, path):
    """Save a dataclass config to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2, default=str))
