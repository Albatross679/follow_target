"""Reward functions for soft manipulator tasks (Choi & Tong, 2025).

Following the Naughton et al. (2021) Elastica benchmark style:
distance-based rewards with optional improvement bonuses and penalties.
"""

import numpy as np


def compute_follow_target_reward(
    tip_pos: np.ndarray,
    target_pos: np.ndarray,
    prev_tip_pos: np.ndarray,
    tip_tangent: np.ndarray | None = None,
    heading_weight: float = 0.0,
    prev_dist: float | None = None,
    pbrs_gamma: float = 0.0,
    pbrs_only: bool = False,
    improvement_weight: float = 0.0,
    return_components: bool = False,
) -> float | tuple[float, dict]:
    """Reward for following a moving target."""
    dist = np.linalg.norm(tip_pos - target_pos)
    dist_reward = np.exp(-5.0 * dist)
    heading_reward = 0.0
    pbrs_reward = 0.0
    improvement_reward = 0.0

    if pbrs_only:
        total = 0.0
    elif heading_weight > 0.0 and tip_tangent is not None:
        to_target = target_pos - tip_pos
        to_target_norm = np.linalg.norm(to_target)
        if to_target_norm > 1e-8:
            to_target_dir = to_target / to_target_norm
            cos_sim = np.dot(tip_tangent, to_target_dir)
            heading_reward = (1.0 + cos_sim) / 2.0
        else:
            heading_reward = 1.0
        total = float((1.0 - heading_weight) * dist_reward + heading_weight * heading_reward)
    else:
        total = float(dist_reward)

    if improvement_weight > 0.0 and prev_dist is not None:
        improvement_reward = float(improvement_weight * (prev_dist - dist))
        total += improvement_reward

    if pbrs_gamma > 0.0 and prev_dist is not None:
        pbrs_reward = float(prev_dist - pbrs_gamma * dist)
        total += pbrs_reward

    if return_components:
        return total, {
            "dist_to_goal": float(dist),
            "reward_dist": float(dist_reward),
            "reward_align": float(heading_reward),
            "reward_pbrs": float(pbrs_reward),
            "reward_improve": float(improvement_reward),
        }
    return total
