"""TorchRL environment for clamped soft manipulator (Choi & Tong, 2025).

Recreates the DisMech-based soft manipulator environment using TorchRL's
EnvBase interface.  When DisMech is not installed, a lightweight mock
physics backend provides the same observation/action/reward interface.
"""

from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    Bounded,
    Composite,
    Unbounded,
)
from torchrl.envs import EnvBase

from choi2025_follow_target.config import Choi2025EnvConfig
from choi2025_follow_target.control import DeltaCurvatureController
from choi2025_follow_target.rewards import compute_follow_target_reward
from choi2025_follow_target.tasks import TargetGenerator

# Try importing DisMech; fall back to mock if unavailable
try:
    import dismech
    from dismech import (
        Environment,
        Geometry,
        GeomParams,
        ImplicitEulerTimeStepper,
        Material,
        SimParams,
        SoftRobot,
    )
    _HAS_DISMECH = True
except ImportError:
    _HAS_DISMECH = False


class _MockRodState:
    """Lightweight mock of DisMech rod state for pipeline validation."""

    def __init__(self, num_nodes: int, segment_length: float, dt: float):
        self._num_nodes = num_nodes
        self._segment_length = segment_length
        self._dt = dt

        self._positions = np.zeros((num_nodes, 3))
        for i in range(num_nodes):
            self._positions[i, 0] = i * segment_length

        self._velocities = np.zeros((num_nodes, 3))

    @property
    def positions(self) -> np.ndarray:
        return self._positions.copy()

    @property
    def velocities(self) -> np.ndarray:
        return self._velocities.copy()

    def apply_curvature_and_step(self, curvature_state: np.ndarray) -> None:
        prev_positions = self._positions.copy()
        n_springs = len(curvature_state)

        for i in range(n_springs):
            kappa1, kappa2 = curvature_state[i]
            angle_xy = kappa1 * self._dt * 0.1
            angle_xz = kappa2 * self._dt * 0.1

            pivot = self._positions[i + 1].copy()
            for j in range(i + 2, self._num_nodes):
                rel = self._positions[j] - pivot
                c, s = np.cos(angle_xy), np.sin(angle_xy)
                new_x = c * rel[0] - s * rel[1]
                new_y = s * rel[0] + c * rel[1]
                c2, s2 = np.cos(angle_xz), np.sin(angle_xz)
                new_x2 = c2 * new_x - s2 * rel[2]
                new_z = s2 * new_x + c2 * rel[2]
                self._positions[j] = pivot + np.array([new_x2, new_y, new_z])

        self._velocities = (self._positions - prev_positions) / self._dt

    def reset(self) -> None:
        for i in range(self._num_nodes):
            self._positions[i] = [i * self._segment_length, 0.0, 0.0]
        self._velocities[:] = 0.0


class SoftManipulatorEnv(EnvBase):
    """Clamped soft manipulator environment for control tasks."""

    def __init__(
        self,
        config: Optional[Choi2025EnvConfig] = None,
        device: str = "cpu",
    ):
        device = torch.device(device) if isinstance(device, str) else device
        super().__init__(device=device, batch_size=torch.Size([]))

        self.config = config or Choi2025EnvConfig()
        self._device = device

        physics = self.config.physics
        self._num_nodes = physics.geometry.num_nodes
        self._num_segments = physics.geometry.num_segments
        self._num_bend_springs = self._num_segments - 1

        self.controller = DeltaCurvatureController(
            num_bend_springs=self._num_bend_springs,
            config=self.config.control,
        )

        self._rng = np.random.default_rng(42)
        self._target = TargetGenerator(self.config.target, self._rng)

        self._use_dismech = _HAS_DISMECH
        self._dismech_robot = None
        self._time_stepper = None
        self._mock_state = None

        self._step_count = 0
        self._prev_tip_pos = np.zeros(3)
        self._prev_dist: float | None = None
        self._reward_components: dict = {}
        self._episode_count = 0

        self._make_spec()

    @property
    def _two_d_sim(self) -> bool:
        return self.config.physics.two_d_sim

    @property
    def _action_dim(self) -> int:
        n_cp = self.config.control.num_control_points
        return n_cp if self._two_d_sim else n_cp * 2

    @property
    def _obs_dim(self) -> int:
        dim = 0
        dim += self._num_nodes * 3  # positions
        dim += self._num_nodes * 3  # velocities
        dim += self._num_bend_springs  # curvatures
        dim += 3  # target position
        return dim

    def _make_spec(self):
        self.observation_spec = Composite(
            observation=Unbounded(
                shape=(self._obs_dim,), dtype=torch.float32, device=self._device
            ),
            dist_to_goal=Unbounded(
                shape=(1,), dtype=torch.float32, device=self._device
            ),
            reward_dist=Unbounded(
                shape=(1,), dtype=torch.float32, device=self._device
            ),
            reward_align=Unbounded(
                shape=(1,), dtype=torch.float32, device=self._device
            ),
            reward_pbrs=Unbounded(
                shape=(1,), dtype=torch.float32, device=self._device
            ),
            reward_improve=Unbounded(
                shape=(1,), dtype=torch.float32, device=self._device
            ),
            shape=(),
        )

        self.action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(self._action_dim,),
            dtype=torch.float32,
            device=self._device,
        )

        self.reward_spec = Unbounded(
            shape=(1,), dtype=torch.float32, device=self._device
        )

        self.done_spec = Composite(
            done=Unbounded(
                shape=(1,), dtype=torch.bool, device=self._device
            ),
            terminated=Unbounded(
                shape=(1,), dtype=torch.bool, device=self._device
            ),
            truncated=Unbounded(
                shape=(1,), dtype=torch.bool, device=self._device
            ),
            shape=(),
        )

    # === DisMech initialization ===

    def _init_dismech(self) -> None:
        physics = self.config.physics

        geom_params = GeomParams(
            rod_r0=physics.snake_radius,
            shell_h=0,
        )

        material = Material(
            density=physics.density,
            youngs_rod=physics.youngs_modulus,
            youngs_shell=0,
            poisson_rod=physics.poisson_ratio,
            poisson_shell=0,
        )

        sim_params = SimParams(
            static_sim=False,
            two_d_sim=physics.two_d_sim,
            use_mid_edge=False,
            use_line_search=True,
            log_data=False,
            log_step=1,
            show_floor=False,
            dt=physics.dt,
            max_iter=physics.max_newton_iter_noncontact,
            total_time=1000.0,
            plot_step=1,
            tol=physics.tol,
            ftol=physics.ftol,
            dtol=physics.dtol,
        )

        env = Environment()

        if physics.enable_gravity:
            env.add_force("gravity", g=np.array(physics.gravity))

        geometry = self._create_rod_geometry()
        self._dismech_robot = SoftRobot(geom_params, material, geometry, sim_params, env)

        if physics.clamp_first_node:
            self._dismech_robot = self._dismech_robot.fix_nodes([0])

        self._time_stepper = ImplicitEulerTimeStepper(self._dismech_robot)

    def _create_rod_geometry(self):
        physics = self.config.physics
        num_nodes = self._num_nodes
        segment_length = physics.snake_length / self._num_segments

        nodes = np.zeros((num_nodes, 3))
        for i in range(num_nodes):
            nodes[i, 0] = i * segment_length

        edges = np.array([[i, i + 1] for i in range(self._num_segments)], dtype=np.int64)
        face_nodes = np.empty((0, 3), dtype=np.int64)

        if _HAS_DISMECH:
            return Geometry(nodes, edges, face_nodes, plot_from_txt=False)
        return nodes, edges, face_nodes

    def _init_mock_physics(self) -> None:
        physics = self.config.physics
        segment_length = physics.snake_length / self._num_segments
        self._mock_state = _MockRodState(
            num_nodes=self._num_nodes,
            segment_length=segment_length,
            dt=physics.dt,
        )

    # === State accessors ===

    def _get_positions(self) -> np.ndarray:
        if self._use_dismech:
            q = self._dismech_robot.state.q
            return q[: 3 * self._num_nodes].reshape(self._num_nodes, 3).copy()
        return self._mock_state.positions

    def _get_velocities(self) -> np.ndarray:
        if self._use_dismech:
            u = self._dismech_robot.state.u
            return u[: 3 * self._num_nodes].reshape(self._num_nodes, 3).copy()
        return self._mock_state.velocities

    def _get_tip_pos(self) -> np.ndarray:
        return self._get_positions()[-1]

    def _get_tip_tangent(self) -> np.ndarray:
        positions = self._get_positions()
        tangent = positions[-1] - positions[-2]
        norm = np.linalg.norm(tangent)
        return tangent / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0])

    def _get_curvatures(self) -> np.ndarray:
        positions = self._get_positions()
        curvatures = []

        for i in range(1, self._num_nodes - 1):
            v1 = positions[i] - positions[i - 1]
            v2 = positions[i + 1] - positions[i]
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm < 1e-8 or v2_norm < 1e-8:
                curvatures.append(0.0)
                continue

            cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            avg_length = (v1_norm + v2_norm) / 2
            curvatures.append(angle / avg_length if avg_length > 1e-8 else 0.0)

        return np.array(curvatures)

    # === Observation construction ===

    def _get_obs(self) -> np.ndarray:
        positions = self._get_positions().flatten()
        velocities = self._get_velocities().flatten()
        curvatures = self._get_curvatures()
        target_pos = self._target.position

        parts = [positions, velocities, curvatures, target_pos]

        return np.concatenate(parts).astype(np.float32)

    # === Apply curvature control ===

    def _apply_curvature(self, curvature_state: np.ndarray) -> None:
        if self._use_dismech:
            bend_springs = self._dismech_robot.bend_springs
            if bend_springs.N > 0 and hasattr(bend_springs, "nat_strain"):
                n = min(len(curvature_state), bend_springs.N)
                for i in range(n):
                    bend_springs.nat_strain[i, 0] = curvature_state[i, 0]
                    bend_springs.nat_strain[i, 1] = curvature_state[i, 1]
        else:
            self._mock_state.apply_curvature_and_step(curvature_state)

    # === EnvBase interface ===

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._target = TargetGenerator(self.config.target, self._rng)

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        if self._use_dismech:
            self._init_dismech()
        else:
            self._init_mock_physics()

        self.controller.reset()

        curriculum = self.config.target.curriculum
        speed_override = None
        if curriculum.enabled:
            progress = min(1.0, self._episode_count / max(1, curriculum.warmup_episodes))
            frac = curriculum.initial_speed_frac + (1.0 - curriculum.initial_speed_frac) * progress
            speed_override = frac * self.config.target.target_speed

        self._target.sample(speed_override=speed_override)

        self._step_count = 0
        self._episode_count += 1
        self._prev_tip_pos = self._get_tip_pos()

        tip_pos = self._get_tip_pos()
        target_pos = self._target.position
        self._prev_dist = float(np.linalg.norm(tip_pos - target_pos))

        obs = self._get_obs()

        _zero = torch.tensor([0.0], dtype=torch.float32, device=self._device)
        return TensorDict(
            {
                "observation": torch.tensor(obs, dtype=torch.float32, device=self._device),
                "done": torch.tensor([False], dtype=torch.bool, device=self._device),
                "terminated": torch.tensor([False], dtype=torch.bool, device=self._device),
                "truncated": torch.tensor([False], dtype=torch.bool, device=self._device),
                "dist_to_goal": _zero.clone(),
                "reward_dist": _zero.clone(),
                "reward_align": _zero.clone(),
                "reward_pbrs": _zero.clone(),
                "reward_improve": _zero.clone(),
            },
            batch_size=self.batch_size,
            device=self._device,
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict["action"].cpu().numpy().astype(np.float64)

        self._prev_tip_pos = self._get_tip_pos()

        curvature_state = self.controller.apply_delta(action, two_d_sim=self._two_d_sim)

        num_substeps = getattr(self.config, 'control_period', 1)
        for substep_i in range(num_substeps):
            if substep_i == 0 or self._use_dismech:
                self._apply_curvature(curvature_state)
            if self._use_dismech:
                try:
                    self._dismech_robot, _ = self._time_stepper.step(
                        self._dismech_robot, debug=False
                    )
                except ValueError:
                    pass

        self._target.step(self.config.physics.dt * num_substeps)

        reward = self._compute_reward()

        self._step_count += 1
        truncated = self._step_count >= self.config.max_episode_steps

        obs = self._get_obs()

        td_dict = {
            "observation": torch.tensor(obs, dtype=torch.float32, device=self._device),
            "reward": torch.tensor([reward], dtype=torch.float32, device=self._device),
            "done": torch.tensor([truncated], dtype=torch.bool, device=self._device),
            "terminated": torch.tensor([False], dtype=torch.bool, device=self._device),
            "truncated": torch.tensor([truncated], dtype=torch.bool, device=self._device),
        }

        for key, val in self._reward_components.items():
            td_dict[key] = torch.tensor([val], dtype=torch.float32, device=self._device)

        return TensorDict(
            td_dict,
            batch_size=self.batch_size,
            device=self._device,
        )

    def _compute_reward(self) -> float:
        tip_pos = self._get_tip_pos()
        target_pos = self._target.position

        tip_tangent = self._get_tip_tangent() if self.config.heading_weight > 0 else None
        reward, self._reward_components = compute_follow_target_reward(
            tip_pos, target_pos, self._prev_tip_pos,
            tip_tangent=tip_tangent,
            heading_weight=self.config.heading_weight,
            prev_dist=self._prev_dist,
            pbrs_gamma=self.config.pbrs_gamma,
            pbrs_only=self.config.pbrs_only,
            improvement_weight=self.config.improvement_weight,
            return_components=True,
        )
        self._prev_dist = float(np.linalg.norm(tip_pos - target_pos))
        return reward

    def close(self, **kwargs):
        self._dismech_robot = None
        self._time_stepper = None
        self._mock_state = None
        super().close(**kwargs)
