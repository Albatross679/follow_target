"""PPO (Proximal Policy Optimization) trainer (self-contained).

A standalone PPO trainer that does not depend on ``src/``.
"""

import json
import os
import signal
import shutil
import tempfile
import time
from collections import deque
from contextlib import nullcontext
from typing import Optional, Dict, Any, Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

from torchrl.envs import EnvBase
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from tqdm import tqdm

from choi2025_follow_target.config import PPOConfig, NetworkConfig, resolve_device, setup_run_dir
from choi2025_follow_target.networks import create_actor, create_critic
from choi2025_follow_target import wandb_utils


STOP_FILE = "STOP"


def _amp_context(use_amp: bool, device: str):
    if use_amp and 'cuda' in str(device):
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return nullcontext()


def compute_explained_variance(value_pred: torch.Tensor, value_target: torch.Tensor) -> float:
    with torch.no_grad():
        y_pred = value_pred.flatten()
        y_true = value_target.flatten()
        var_y = y_true.var()
        if var_y < 1e-8:
            return 0.0
        return (1.0 - (y_true - y_pred).var() / var_y).item()


def collect_system_metrics(device, counter: list, interval: int) -> dict:
    counter[0] += 1
    if counter[0] % interval != 0:
        return {}
    metrics = {}
    try:
        import psutil
        metrics["system/cpu_percent"] = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        metrics["system/ram_percent"] = mem.percent
    except ImportError:
        pass
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        dev = torch.device(device)
        metrics["system/gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(dev) / (1024**2)
    return metrics


class PPOTrainer:
    """PPO trainer for continuous control tasks."""

    def __init__(
        self,
        env: EnvBase,
        config: Optional[PPOConfig] = None,
        network_config: Optional[NetworkConfig] = None,
        device: str = "cpu",
        run_dir: Optional[Path] = None,
    ):
        self.env = env
        self.config = config or PPOConfig()
        self.network_config = network_config or NetworkConfig()
        self.device = resolve_device(device)

        obs_dim = env.observation_spec["observation"].shape[-1]
        action_spec = env.action_spec

        self.actor = create_actor(
            obs_dim=obs_dim,
            action_spec=action_spec,
            config=self.network_config.actor,
            device=self.device,
        )

        self.critic = create_critic(
            obs_dim=obs_dim,
            config=self.network_config.critic,
            device=self.device,
        )

        self.loss_module = ClipPPOLoss(
            actor_network=self.actor,
            critic_network=self.critic,
            clip_epsilon=self.config.clip_epsilon,
            entropy_coeff=self.config.entropy_coef,
            critic_coeff=self.config.value_coef,
            normalize_advantage=self.config.normalize_advantage,
        )

        self.advantage_module = GAE(
            gamma=self.config.gamma,
            lmbda=self.config.gae_lambda,
            value_network=self.critic,
        )

        self.optimizer = Adam(
            self.loss_module.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        if self.config.lr_schedule == "linear":
            total_updates = max(1, self.config.total_frames // self.config.frames_per_batch)
            self.scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: max(
                    self.config.lr_end / self.config.learning_rate,
                    1.0 - step / total_updates,
                ),
            )
        else:
            self.scheduler = None

        self.collector = SyncDataCollector(
            create_env_fn=lambda: env,
            policy=self.actor,
            frames_per_batch=self.config.frames_per_batch,
            total_frames=self.config.total_frames,
            device=self.device,
        )

        # Training state
        self.total_frames = 0
        self.total_episodes = 0
        self.best_reward = float("-inf")
        self._episode_reward_buffer = deque(maxlen=100)
        self._batches_since_improvement = 0
        self._patience_batches = self.config.patience_batches
        self._train_start_time = 0.0
        self._batch_start_time = 0.0
        self._system_log_counter = [0]

        # Graceful shutdown
        self._shutdown_requested = False
        self._original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, self._signal_handler)

        # Output directories
        if run_dir is None:
            run_dir = setup_run_dir(self.config)
        self.run_dir = Path(run_dir)
        self.log_dir = self.run_dir
        self.save_dir = self.run_dir / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._metrics_log_path = self.run_dir / "metrics.jsonl"
        self._metrics_log_file = open(self._metrics_log_path, "a", encoding="utf-8")

        # W&B
        self.wandb_run = wandb_utils.setup_run(self.config, self.run_dir)
        extra_params = wandb_utils._count_parameters(self.loss_module)
        extra_params["num_envs"] = self.config.num_envs
        extra_params["use_amp"] = self.config.use_amp
        extra_params.update(wandb_utils.collect_hardware_info(self.device))
        wandb_utils.log_extra_params(self.wandb_run, extra_params)

    def _signal_handler(self, signum: int, frame) -> None:
        signal_name = signal.Signals(signum).name
        print(f"\n{signal_name} received, requesting graceful shutdown...")
        self._shutdown_requested = True

    def _restore_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._original_sigint_handler)
        signal.signal(signal.SIGTERM, self._original_sigterm_handler)

    def _check_stop_file(self) -> bool:
        return Path(STOP_FILE).exists()

    def _write_metrics_jsonl(self, metrics: Dict[str, Any]) -> None:
        line = json.dumps(metrics, default=str)
        self._metrics_log_file.write(line + "\n")
        self._metrics_log_file.flush()

    def train(
        self, callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run PPO training loop."""
        pbar = tqdm(total=self.config.total_frames, desc="Training")
        all_metrics = []

        self._train_start_time = time.monotonic()
        self._batch_start_time = time.monotonic()
        max_wall_time = self.config.max_wall_time
        stop_reason = "completed"

        try:
            _collector_start = time.monotonic()

            for batch_idx, batch in enumerate(self.collector):
                env_dt = time.monotonic() - _collector_start

                if self._shutdown_requested:
                    print("Shutdown requested, saving checkpoint...")
                    self.save_checkpoint("interrupted")
                    self._restore_signal_handlers()
                    stop_reason = "signal"
                    break

                if self._check_stop_file():
                    tqdm.write("STOP file detected. Stopping.")
                    stop_reason = "stop_file"
                    break

                if max_wall_time is not None:
                    elapsed = time.monotonic() - self._train_start_time
                    if elapsed >= max_wall_time:
                        tqdm.write(f"Wall-clock limit reached ({elapsed:.0f}s). Stopping.")
                        stop_reason = "wall_time"
                        break

                if self._patience_batches > 0 and self._batches_since_improvement >= self._patience_batches:
                    tqdm.write(f"Early stopping: no improvement for {self._batches_since_improvement} batches.")
                    stop_reason = "early_stopping"
                    break

                t0 = time.monotonic()
                batch = batch.to(self.device)
                if batch.ndim > 1:
                    batch = batch.reshape(-1)

                with torch.no_grad():
                    self.advantage_module(batch)
                data_dt = time.monotonic() - t0

                t0 = time.monotonic()
                metrics = self._update(batch)
                backward_dt = time.monotonic() - t0

                metrics["batch_idx"] = batch_idx
                metrics["total_frames"] = self.total_frames

                self.total_frames += batch.numel()
                pbar.update(batch.numel())

                # Episode statistics
                next_td = batch.get("next", batch)
                done_mask = next_td["done"].squeeze(-1)

                if "episode_reward" in next_td.keys():
                    episode_rewards = next_td["episode_reward"][done_mask]
                    if len(episode_rewards) > 0:
                        metrics["mean_episode_reward"] = episode_rewards.mean().item()
                        metrics["max_episode_reward"] = episode_rewards.max().item()
                        metrics["min_episode_reward"] = episode_rewards.min().item()
                        self.total_episodes += len(episode_rewards)

                        for r in episode_rewards.tolist():
                            self._episode_reward_buffer.append(r)

                        if len(self._episode_reward_buffer) > 0:
                            metrics["rolling_mean_reward_100"] = float(
                                np.mean(list(self._episode_reward_buffer))
                            )

                        if metrics["mean_episode_reward"] > self.best_reward:
                            self.best_reward = metrics["mean_episode_reward"]
                            self._batches_since_improvement = 0
                            self.save_checkpoint("best")
                        else:
                            self._batches_since_improvement += 1

                if "step_count" in next_td.keys():
                    episode_lengths = next_td["step_count"][done_mask]
                    if len(episode_lengths) > 0:
                        metrics["mean_episode_length"] = episode_lengths.float().mean().item()

                # Reward diagnostics
                for key in ("dist_to_goal", "reward_dist", "reward_align", "reward_pbrs"):
                    if key in next_td.keys():
                        metrics[f"mean_{key}"] = next_td[key].mean().item()

                metrics["total_episodes"] = self.total_episodes
                metrics["best_reward"] = self.best_reward
                metrics["batches_since_improvement"] = self._batches_since_improvement
                batch_time_s = time.monotonic() - self._batch_start_time
                metrics["fps"] = batch.numel() / batch_time_s if batch_time_s > 0 else 0.0
                self._batch_start_time = time.monotonic()

                t0 = time.monotonic()
                all_metrics.append(metrics)
                self._write_metrics_jsonl(metrics)

                if batch_idx % self.config.log_interval == 0:
                    self._log_metrics(metrics)

                if batch_idx % self.config.save_interval == 0:
                    self.save_checkpoint(f"step_{self.total_frames}")

                if callback:
                    callback(metrics)

                if self.scheduler:
                    self.scheduler.step()

                _collector_start = time.monotonic()

        finally:
            pbar.close()
            self._metrics_log_file.close()

        self.save_checkpoint("final")

        best_path = self.save_dir / "best.pt"
        if best_path.exists():
            wandb_utils.log_model_artifact(
                self.wandb_run, best_path, artifact_name=self.config.name,
                metadata={
                    "best_reward": self.best_reward,
                    "total_frames": self.total_frames,
                    "stop_reason": stop_reason,
                },
            )

        wandb_utils.end_run(self.wandb_run)

        return {
            "total_frames": self.total_frames,
            "total_episodes": self.total_episodes,
            "best_reward": self.best_reward,
            "stop_reason": stop_reason,
            "metrics": all_metrics,
        }

    def _update(self, batch: TensorDict) -> Dict[str, float]:
        """Perform PPO update on batch."""
        metrics = {
            "loss_actor": 0.0,
            "loss_critic": 0.0,
            "loss_entropy": 0.0,
            "kl_divergence": 0.0,
            "grad_norm": 0.0,
            "clip_fraction": 0.0,
        }

        # Pre-update diagnostics
        with torch.no_grad():
            advantages = batch["advantage"]
            metrics["advantage_mean"] = advantages.mean().item()
            metrics["advantage_std"] = advantages.std().item()

            if "value_target" in batch.keys() and "state_value" in batch.keys():
                metrics["explained_variance"] = compute_explained_variance(
                    batch["state_value"], batch["value_target"],
                )

        kl_break = False
        actual_updates = 0
        for epoch in range(self.config.num_epochs):
            indices = torch.randperm(batch.numel())
            num_batches = max(1, batch.numel() // self.config.mini_batch_size)

            for i in range(num_batches):
                start = i * self.config.mini_batch_size
                end = min((i + 1) * self.config.mini_batch_size, batch.numel())
                mb_indices = indices[start:end]
                mini_batch = batch[mb_indices]

                loss_dict = self.loss_module(mini_batch)

                loss = (
                    loss_dict["loss_objective"]
                    + self.config.value_coef * loss_dict["loss_critic"]
                    + self.config.entropy_coef * loss_dict.get("loss_entropy", 0.0)
                )

                self.optimizer.zero_grad()

                if not torch.isfinite(loss):
                    continue

                loss.backward()

                if self.config.max_grad_norm > 0:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.loss_module.parameters(), self.config.max_grad_norm,
                    )
                else:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.loss_module.parameters(), float('inf'),
                    )

                if grad_norm is not None and not torch.isfinite(grad_norm):
                    self.optimizer.zero_grad()
                    continue

                self.optimizer.step()
                actual_updates += 1

                metrics["loss_actor"] += loss_dict["loss_objective"].item()
                metrics["loss_critic"] += loss_dict["loss_critic"].item()
                if "loss_entropy" in loss_dict:
                    metrics["loss_entropy"] += loss_dict["loss_entropy"].item()
                if "kl_approx" in loss_dict:
                    metrics["kl_divergence"] += loss_dict["kl_approx"].item()
                if grad_norm is not None:
                    metrics["grad_norm"] += float(grad_norm)
                if "clip_fraction" in loss_dict:
                    metrics["clip_fraction"] += loss_dict["clip_fraction"].item()

                if self.config.target_kl and "kl_approx" in loss_dict:
                    batch_kl = loss_dict["kl_approx"].item()
                    if batch_kl > 1.5 * self.config.target_kl:
                        kl_break = True
                        break

            if kl_break:
                break
            avg_kl = metrics["kl_divergence"] / max(1, actual_updates)
            if self.config.target_kl and avg_kl > self.config.target_kl:
                break

        _accumulated_keys = {
            "loss_actor", "loss_critic", "loss_entropy",
            "kl_divergence", "grad_norm", "clip_fraction",
        }
        for key in _accumulated_keys:
            metrics[key] /= max(1, actual_updates)

        return metrics

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log training metrics to console and W&B."""
        log_str = f"Step {self.total_frames}: "
        log_str += f"actor_loss={metrics.get('loss_actor', 0):.4f}, "
        log_str += f"critic_loss={metrics.get('loss_critic', 0):.4f}, "
        log_str += f"kl={metrics['kl_divergence']:.4f}"

        if "mean_episode_reward" in metrics:
            log_str += f", reward={metrics['mean_episode_reward']:.2f}"
        if "rolling_mean_reward_100" in metrics:
            log_str += f", rolling100={metrics['rolling_mean_reward_100']:.2f}"

        tqdm.write(log_str)

        wandb_log = {
            "train/actor_loss": metrics["loss_actor"],
            "train/critic_loss": metrics["loss_critic"],
            "train/entropy_loss": metrics["loss_entropy"],
            "train/kl_divergence": metrics["kl_divergence"],
            "train/clip_fraction": metrics.get("clip_fraction", 0.0),
            "gradients/grad_norm": metrics.get("grad_norm", 0.0),
        }

        if "mean_episode_reward" in metrics:
            wandb_log["episode/mean_reward"] = metrics["mean_episode_reward"]
        if "rolling_mean_reward_100" in metrics:
            wandb_log["episode/rolling_mean_reward_100"] = metrics["rolling_mean_reward_100"]
        if "mean_episode_length" in metrics:
            wandb_log["episode/mean_length"] = metrics["mean_episode_length"]
        if "total_episodes" in metrics:
            wandb_log["episode/count"] = metrics["total_episodes"]

        wandb_log["tracking/best_reward"] = self.best_reward

        if self.scheduler:
            wandb_log["train/learning_rate"] = self.scheduler.get_last_lr()[0]

        if "fps" in metrics:
            wandb_log["timing/fps"] = metrics["fps"]
        wandb_log["timing/wall_clock_mins"] = (time.monotonic() - self._train_start_time) / 60.0

        # Reward diagnostics
        for key in ("mean_dist_to_goal", "mean_reward_dist", "mean_reward_align", "mean_reward_pbrs"):
            if key in metrics:
                wandb_log[f"reward/{key.replace('mean_', '')}"] = metrics[key]

        if "explained_variance" in metrics:
            wandb_log["diagnostics/explained_variance"] = metrics["explained_variance"]

        sys_metrics = collect_system_metrics(self.device, self._system_log_counter, 10)
        wandb_log.update(sys_metrics)

        wandb_utils.log_metrics(self.wandb_run, wandb_log, step=self.total_frames)

    def save_checkpoint(self, name: str) -> None:
        """Save training checkpoint atomically."""
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_frames": self.total_frames,
            "total_episodes": self.total_episodes,
            "best_reward": self.best_reward,
            "config": self.config,
        }

        path = self.save_dir / f"{name}.pt"

        if path.exists():
            backup_path = self.save_dir / f"{name}.pt.backup"
            shutil.copy2(path, backup_path)

        fd, temp_path = tempfile.mkstemp(dir=self.save_dir, suffix='.pt.tmp')
        try:
            torch.save(checkpoint, temp_path)
            os.rename(temp_path, path)
        except Exception:
            os.unlink(temp_path)
            raise
        finally:
            os.close(fd)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_frames = checkpoint["total_frames"]
        self.total_episodes = checkpoint["total_episodes"]
        self.best_reward = checkpoint["best_reward"]
