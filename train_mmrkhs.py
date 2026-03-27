"""Training script for soft manipulator MM-RKHS (Choi & Tong, 2025 setup).

Self-contained — runs without any dependency on ``src/``.

Usage:
    python -m choi2025_follow_target.train_mmrkhs --total-frames 1000000
    python -m choi2025_follow_target.train_mmrkhs --seed 0 --num-envs 32
    python -m choi2025_follow_target.train_mmrkhs --max-wall-time 30m
"""

import argparse
import os

# Limit thread spawning for parallel envs to avoid pthread exhaustion
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from torchrl.envs import RewardSum

from choi2025_follow_target.config import (
    Choi2025MMRKHSConfig,
    Choi2025EnvConfig,
    resolve_device,
    setup_run_dir,
)
from choi2025_follow_target.env import SoftManipulatorEnv
from choi2025_follow_target.train_sac import parse_wall_time
from choi2025_follow_target.trainer_mmrkhs import MMRKHSTrainer


def _make_env(env_config, device):
    """Top-level env factory for picklability with ParallelEnv."""
    return SoftManipulatorEnv(env_config, device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train soft manipulator MM-RKHS (Gupta & Mahajan)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument(
        "--total-frames", type=int, default=None, help="Total training frames"
    )
    parser.add_argument(
        "--num-envs", type=int, default=32, help="Number of parallel envs"
    )
    parser.add_argument(
        "--max-wall-time",
        type=str,
        default=None,
        help="Wall-clock time limit, e.g. '30m', '2h', '1h30m', or '3600' (seconds)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = resolve_device(args.device)

    env_config = Choi2025EnvConfig(device=device)
    config = Choi2025MMRKHSConfig(seed=args.seed, device=device, env=env_config)

    if args.total_frames is not None:
        config.total_frames = args.total_frames
    if args.max_wall_time is not None:
        config.max_wall_time = parse_wall_time(args.max_wall_time)
    if args.num_envs > 1:
        config.num_envs = args.num_envs

    config.__post_init__()

    run_dir = setup_run_dir(config)

    # Create environment
    if config.num_envs > 1:
        from torchrl.envs import ParallelEnv

        env = ParallelEnv(
            num_workers=config.num_envs,
            create_env_fn=[
                lambda cfg=env_config: _make_env(cfg, "cpu")
            ] * config.num_envs,
        )
    else:
        env = SoftManipulatorEnv(config.env, device=device)

    env = env.append_transform(RewardSum())

    try:
        trainer = MMRKHSTrainer(
            env=env,
            config=config,
            network_config=config.network,
            device=device,
            run_dir=run_dir,
        )

        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
            print(f"  Resumed at frame {trainer.total_frames}, best_reward={trainer.best_reward:.2f}")

        wall_msg = ""
        if config.max_wall_time is not None:
            mins = config.max_wall_time / 60
            wall_msg = f", max wall time {mins:.0f}min"
        print(f"Training follow_target with MM-RKHS, {config.total_frames} frames{wall_msg}")
        print(f"  Device: {device}")
        print(f"  Run directory: {run_dir}")
        results = trainer.train()
        print(f"Done: {results['total_episodes']} episodes, best={results['best_reward']:.2f}")
    finally:
        try:
            env.close()
        except RuntimeError:
            pass


if __name__ == "__main__":
    main()
