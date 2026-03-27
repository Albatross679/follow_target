"""MM-RKHS (Gupta & Mahajan) trainer (self-contained).

Implements the MM-RKHS algorithm from Gupta & Mahajan (2026, arXiv:2603.17875).

Key differences from PPO:
- No ClipPPOLoss -- custom loss computed directly
- Trust region via MMD penalty + KL regularizer (no clipping)
- No entropy bonus -- KL regularizer handles exploration
- MMD via linear-time RBF kernel estimator
- Old distribution reconstructed from stored loc/scale in batch
"""

import json
import math
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
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

from torchrl.envs import EnvBase
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.modules import TanhNormal
from tensordict import TensorDict
from tqdm import tqdm

from choi2025_follow_target.config import MMRKHSConfig, NetworkConfig, resolve_device, setup_run_dir
from choi2025_follow_target.networks import create_actor, create_critic
from choi2025_follow_target import wandb_utils


STOP_FILE = "STOP"


def _amp_context(use_amp: bool, device: str):
    if use_amp and 'cuda' in str(device):
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return nullcontext()


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


class MMRKHSTrainer:
    """MM-RKHS trainer for continuous control tasks.

    Loss = -E[ratio * A] + beta * MMD^2 + (1/eta) * KL + value_coef * critic_loss
    """

    def __init__(
        self,
        env: EnvBase,
        config: Optional[MMRKHSConfig] = None,
        network_config: Optional[NetworkConfig] = None,
        device: str = "cpu",
        run_dir: Optional[Path] = None,
    ):
        self.env = env
        self.config = config or MMRKHSConfig()
        self.network_config = network_config or NetworkConfig()
        self.device = resolve_device(device)

        obs_dim = env.observation_spec["observation"].shape[-1]
        action_spec = env.action_spec
        self.action_spec = action_spec

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

        self.loss_params = list(self.actor.parameters()) + list(self.critic.parameters())

        self.optimizer = Adam(
            self.loss_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.advantage_module = GAE(
            gamma=self.config.gamma,
            lmbda=self.config.gae_lambda,
            value_network=self.critic,
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
        self._global_batch_idx = 0
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
        total_params = sum(p.numel() for p in self.loss_params)
        trainable_params = sum(p.numel() for p in self.loss_params if p.requires_grad)
        extra_params = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "num_envs": self.config.num_envs,
            "use_amp": self.config.use_amp,
            "algorithm": "MM-RKHS",
            "beta": self.config.beta,
            "eta": self.config.eta,
            "mmd_bandwidth": self.config.mmd_bandwidth,
        }
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

    def _compute_mmd_penalty(
        self, obs, old_loc, old_scale, new_loc, new_scale,
    ) -> torch.Tensor:
        """Compute MMD^2 penalty via linear-time unbiased RBF kernel estimator."""
        num_samples = self.config.mmd_num_samples
        bandwidth = self.config.mmd_bandwidth

        action_low = self.action_spec.space.low.to(old_loc.device)
        action_high = self.action_spec.space.high.to(old_loc.device)

        old_dist = TanhNormal(old_loc.detach(), old_scale.detach(), low=action_low, high=action_high)
        new_dist = TanhNormal(new_loc, new_scale, low=action_low, high=action_high)

        with torch.no_grad():
            x = old_dist.rsample((num_samples,))
        y = new_dist.rsample((num_samples,))

        n_pairs = num_samples // 2
        if n_pairs < 1:
            return torch.tensor(0.0, device=old_loc.device, requires_grad=True)

        x1, x2 = x[:n_pairs], x[n_pairs:2 * n_pairs]
        y1, y2 = y[:n_pairs], y[n_pairs:2 * n_pairs]

        def rbf_kernel(a, b):
            diff = (a - b).pow(2).sum(dim=-1)
            return torch.exp(-diff / (2.0 * bandwidth ** 2))

        k_xx = rbf_kernel(x1, x2)
        k_yy = rbf_kernel(y1, y2)
        k_xy = rbf_kernel(x1, y2)
        k_yx = rbf_kernel(x2, y1)

        mmd_sq = (k_xx + k_yy - k_xy - k_yx).mean()
        return mmd_sq.clamp(min=0.0)

    def train(
        self, callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run MM-RKHS training loop."""
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
                self._global_batch_idx += 1
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

                for key in ("dist_to_goal", "reward_dist", "reward_align"):
                    if key in next_td.keys():
                        metrics[f"mean_{key}"] = next_td[key].mean().item()

                metrics["total_episodes"] = self.total_episodes
                metrics["best_reward"] = self.best_reward
                metrics["batches_since_improvement"] = self._batches_since_improvement
                batch_time_s = time.monotonic() - self._batch_start_time
                metrics["fps"] = batch.numel() / batch_time_s if batch_time_s > 0 else 0.0
                self._batch_start_time = time.monotonic()

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
        """Perform MM-RKHS update on batch."""
        metrics = {
            "loss_policy": 0.0,
            "loss_critic": 0.0,
            "mmd_penalty": 0.0,
            "kl_divergence": 0.0,
            "grad_norm": 0.0,
            "policy_entropy": 0.0,
            "kernel_correction": 0.0,
        }

        # Adaptive eta
        if self.config.eta_schedule:
            k = self._global_batch_idx
            eta_effective = self.config.eta * (k + 1) ** self.config.eta_exponent
        else:
            eta_effective = self.config.eta

        actual_updates = 0
        beta_effective = self.config.beta

        for epoch in range(self.config.num_epochs):
            indices = torch.randperm(batch.numel())
            num_batches = max(1, batch.numel() // self.config.mini_batch_size)

            for i in range(num_batches):
                start = i * self.config.mini_batch_size
                end = min((i + 1) * self.config.mini_batch_size, batch.numel())
                mb_indices = indices[start:end]
                mb = batch[mb_indices]

                log_prob_old = mb["action_log_prob"].detach()
                old_loc = mb["loc"].detach()
                old_scale = mb["scale"].detach()

                advantage = mb["advantage"]
                if self.config.normalize_advantage:
                    adv_mean = advantage.mean()
                    adv_std = advantage.std() + 1e-8
                    advantage = (advantage - adv_mean) / adv_std

                # Adaptive beta
                if self.config.beta_schedule:
                    k = self._global_batch_idx
                    beta_effective = advantage.abs().max().item() / max(math.sqrt(k + 1), 1.0)
                else:
                    beta_effective = self.config.beta

                # Inner MM iterations
                for mm_iter in range(self.config.inner_mm_iterations):
                    td_fwd = self.actor(mb.clone())
                    log_prob_new = td_fwd["action_log_prob"]
                    new_loc = td_fwd["loc"]
                    new_scale = td_fwd["scale"]

                    log_ratio = (log_prob_new - log_prob_old).clamp(
                        -self.config.exponent_clip, self.config.exponent_clip
                    )
                    ratio = log_ratio.exp()

                    surr_advantage = ratio * advantage

                    mmd = self._compute_mmd_penalty(
                        obs=mb["observation"],
                        old_loc=old_loc, old_scale=old_scale,
                        new_loc=new_loc, new_scale=new_scale,
                    )

                    kl = (ratio - 1.0 - log_ratio).mean()

                    value_pred = self.critic(mb)["state_value"]
                    critic_loss = F.mse_loss(value_pred, mb["value_target"])

                    with torch.no_grad():
                        entropy = 0.5 * torch.log(2 * math.pi * math.e * new_scale.pow(2)).mean()

                    if self.config.kernel_correction:
                        loc_diff = old_loc - new_loc
                        kernel_corr = (loc_diff.pow(2).sum(dim=-1) * advantage.abs()).mean()
                        kernel_corr = self.config.kernel_correction_weight * kernel_corr
                    else:
                        kernel_corr = torch.tensor(0.0, device=batch.device)

                    total_loss = (
                        -surr_advantage.mean()
                        + beta_effective * mmd
                        + (1.0 / eta_effective) * kl
                        + self.config.value_coef * critic_loss
                        + kernel_corr
                    )

                    self.optimizer.zero_grad()

                    if not torch.isfinite(total_loss):
                        continue

                    total_loss.backward()

                    if self.config.max_grad_norm > 0:
                        grad_norm = nn.utils.clip_grad_norm_(self.loss_params, self.config.max_grad_norm)
                    else:
                        grad_norm = nn.utils.clip_grad_norm_(self.loss_params, float('inf'))

                    if not torch.isfinite(grad_norm):
                        self.optimizer.zero_grad()
                        continue

                    self.optimizer.step()
                    actual_updates += 1

                    metrics["loss_policy"] += (-surr_advantage.mean()).item()
                    metrics["loss_critic"] += critic_loss.item()
                    metrics["mmd_penalty"] += mmd.item()
                    metrics["kl_divergence"] += kl.item()
                    metrics["grad_norm"] += float(grad_norm)
                    metrics["policy_entropy"] += entropy.item()
                    metrics["kernel_correction"] += kernel_corr.item()

        accumulated_keys = {
            "loss_policy", "loss_critic", "mmd_penalty",
            "kl_divergence", "grad_norm", "policy_entropy",
            "kernel_correction",
        }
        for key in accumulated_keys:
            metrics[key] /= max(1, actual_updates)

        metrics["eta_effective"] = eta_effective
        metrics["beta_effective"] = beta_effective

        return metrics

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        log_str = f"Step {self.total_frames}: "
        log_str += f"policy_loss={metrics['loss_policy']:.4f}, "
        log_str += f"critic_loss={metrics['loss_critic']:.4f}, "
        log_str += f"mmd={metrics['mmd_penalty']:.4f}, "
        log_str += f"kl={metrics['kl_divergence']:.4f}"

        if "mean_episode_reward" in metrics:
            log_str += f", reward={metrics['mean_episode_reward']:.2f}"
        if "rolling_mean_reward_100" in metrics:
            log_str += f", rolling100={metrics['rolling_mean_reward_100']:.2f}"

        tqdm.write(log_str)

        wandb_log = {
            "train/actor_loss": metrics["loss_policy"],
            "train/critic_loss": metrics["loss_critic"],
            "train/mmd_penalty": metrics["mmd_penalty"],
            "train/kl_divergence": metrics["kl_divergence"],
            "train/policy_entropy": metrics.get("policy_entropy", 0.0),
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

        if "eta_effective" in metrics:
            wandb_log["train/eta_effective"] = metrics["eta_effective"]
        if "beta_effective" in metrics:
            wandb_log["train/beta_effective"] = metrics["beta_effective"]
        if "kernel_correction" in metrics:
            wandb_log["train/kernel_correction"] = metrics["kernel_correction"]

        wandb_log["tracking/best_reward"] = self.best_reward

        if self.scheduler:
            wandb_log["train/learning_rate"] = self.scheduler.get_last_lr()[0]

        if "fps" in metrics:
            wandb_log["timing/fps"] = metrics["fps"]
        wandb_log["timing/wall_clock_mins"] = (time.monotonic() - self._train_start_time) / 60.0

        for key in ("mean_dist_to_goal", "mean_reward_dist", "mean_reward_align"):
            if key in metrics:
                wandb_log[f"reward/{key.replace('mean_', '')}"] = metrics[key]

        sys_metrics = collect_system_metrics(self.device, self._system_log_counter, 10)
        wandb_log.update(sys_metrics)

        wandb_utils.log_metrics(self.wandb_run, wandb_log, step=self.total_frames)

    def save_checkpoint(self, name: str) -> None:
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
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_frames = checkpoint["total_frames"]
        self.total_episodes = checkpoint["total_episodes"]
        self.best_reward = checkpoint["best_reward"]
