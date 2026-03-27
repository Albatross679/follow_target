"""Delta curvature controller with Voronoi smoothing (Choi & Tong, 2025).

The agent outputs delta natural curvature at `num_control_points` uniformly
spaced locations along the rod.  These are interpolated to all bend springs
via a Voronoi weight matrix, yielding a smooth curvature distribution.
"""

import numpy as np

from choi2025_follow_target.config import DeltaCurvatureControlConfig


class DeltaCurvatureController:
    """Controls rod curvature via delta commands at sparse control points."""

    def __init__(
        self,
        num_bend_springs: int,
        config: DeltaCurvatureControlConfig | None = None,
    ):
        self.config = config or DeltaCurvatureControlConfig()
        self.num_bend_springs = num_bend_springs
        self.num_control_points = self.config.num_control_points

        self.W = self._build_voronoi_weights()
        self.curvature_state = np.zeros((num_bend_springs, 2))

    def _build_voronoi_weights(self) -> np.ndarray:
        """Build Voronoi interpolation weight matrix."""
        n_springs = self.num_bend_springs
        n_cp = self.num_control_points

        if not self.config.voronoi_smoothing or n_cp >= n_springs:
            W = np.zeros((n_springs, n_cp))
            for i in range(n_springs):
                cp_idx = min(int(i * n_cp / n_springs), n_cp - 1)
                W[i, cp_idx] = 1.0
            return W

        spring_pos = np.linspace(0.0, 1.0, n_springs)
        cp_pos = np.linspace(0.0, 1.0, n_cp)

        W = np.zeros((n_springs, n_cp))

        for i, s in enumerate(spring_pos):
            right_idx = np.searchsorted(cp_pos, s)

            if right_idx == 0:
                W[i, 0] = 1.0
            elif right_idx >= n_cp:
                W[i, n_cp - 1] = 1.0
            else:
                left_idx = right_idx - 1
                left_pos = cp_pos[left_idx]
                right_pos = cp_pos[right_idx]
                span = right_pos - left_pos

                if span < 1e-12:
                    W[i, left_idx] = 1.0
                else:
                    alpha = (s - left_pos) / span
                    W[i, left_idx] = 1.0 - alpha
                    W[i, right_idx] = alpha

        return W

    def apply_delta(self, action: np.ndarray, two_d_sim: bool = False) -> np.ndarray:
        """Apply delta curvature action and return updated curvature state."""
        max_delta = self.config.max_delta_curvature
        n_cp = self.num_control_points

        if two_d_sim:
            delta_cp = np.clip(action[:n_cp], -1.0, 1.0) * max_delta
            delta_springs = self.W @ delta_cp
            self.curvature_state[:, 0] += delta_springs
        else:
            delta_cp1 = np.clip(action[:n_cp], -1.0, 1.0) * max_delta
            delta_cp2 = np.clip(action[n_cp : 2 * n_cp], -1.0, 1.0) * max_delta
            self.curvature_state[:, 0] += self.W @ delta_cp1
            self.curvature_state[:, 1] += self.W @ delta_cp2

        return self.curvature_state.copy()

    def reset(self) -> None:
        """Reset curvature state to zero."""
        self.curvature_state[:] = 0.0
