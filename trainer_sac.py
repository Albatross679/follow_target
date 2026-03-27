"""SAC (Soft Actor-Critic) trainer (self-contained).

A standalone SAC trainer that does not depend on ``src/``.
"""

import json
import os
import signal
import time
from contextlib import nullcontext
from typing import Optional, Dict, Any, Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from torchrl.envs import EnvBase
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from tqdm import tqdm

from choi2025_follow_target.config import SACConfig, NetworkConfig, resolve_device, setup_run_dir
from choi2025_follow_target.networks import create_actor, TwinQNetwork
from choi2025_follow_target import wandb_utils


# Default STOP file path
STOP_FILE = "STOP"


def _amp_context(use_amp: bool, device: str):
    if use_amp and 'cuda' in str(device):
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return nullcontext()


def compute_grad_norm(model: nn.Module) -> float:
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    return total_norm_sq ** 0.5


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
        metrics["system/ram_used_gb"] = mem.used / (1024**3)
    except ImportError:
        pass
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        dev = torch.device(device)
        metrics["system/gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(dev) / (1024**2)
        metrics["system/gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(dev) / (1024**2)
    return metrics


class SACTrainer:
    """SAC trainer for continuous control tasks."""

    def __init__(
        self,
        env: EnvBase,
        config: Optional[SACConfig] = None,
        network_config: Optional[NetworkConfig] = None,
        device: str = "cpu",
        run_dir: Optional[Path] = None,
    ):
        self.env = env
        self.config = config or SACConfig()
        self.network_config = network_config or NetworkConfig()
        self.device = resolve_device(device)

        obs_dim = env.observation_spec["observation"].shape[-1]
        action_spec = env.action_spec
        action_dim = action_spec.shape[-1]

        # Create actor
        self.actor = create_actor(
            obs_dim=obs_dim,
            action_spec=action_spec,
            config=self.network_config.actor,
            device=self.device,
        )

        # Create twin Q-networks
        self.critic = TwinQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=self.network_config.critic.hidden_dims,
            activation=self.network_config.critic.activation,
        ).to(self.device)

        self.critic_target = TwinQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=self.network_config.critic.hidden_dims,
            activation=self.network_config.critic.activation,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Entropy coefficient
        self._use_entropy = self.config.auto_alpha or self.config.alpha > 0.0
        if self.config.auto_alpha:
            self.log_alpha = torch.tensor(
                np.log(self.config.alpha), requires_grad=True, device=self.device
            )
            self.target_entropy = (
                self.config.target_entropy
                if self.config.target_entropy is not None
                else -action_dim
            )
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.config.alpha_lr)
        else:
            safe_alpha = max(self.config.alpha, 1e-10)
            self.log_alpha = torch.tensor(np.log(safe_alpha), device=self.device)

        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.config.critic_lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.config.buffer_size, device=self.device),
            batch_size=self.config.batch_size,
        )

        # Training state
        self.total_frames = 0
        self.total_episodes = 0
        self.best_reward = float("-inf")
        self._update_count = 0
        self._train_start_time = 0.0
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

        # Metrics log
        self._metrics_log_path = self.run_dir / "metrics.jsonl"
        self._metrics_log_file = open(self._metrics_log_path, "a", encoding="utf-8")

        # W&B
        self.wandb_run = wandb_utils.setup_run(self.config, self.run_dir)

        extra_params = wandb_utils._count_parameters(self.actor)
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

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def train(
        self, callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run SAC training loop."""
        is_vec = len(self.env.batch_size) > 0
        num_envs = self.env.batch_size[0] if is_vec else 1

        pbar = tqdm(total=self.config.total_frames, desc="Training")
        all_metrics = []

        start_time = time.monotonic()
        self._train_start_time = start_time
        max_wall_time = self.config.max_wall_time
        stop_reason = "completed"

        episode_rewards = np.zeros(num_envs)
        episode_lengths = np.zeros(num_envs, dtype=int)

        td = self.env.reset()

        try:
            while self.total_frames < self.config.total_frames:
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
                    elapsed = time.monotonic() - start_time
                    if elapsed >= max_wall_time:
                        tqdm.write(f"Wall-clock limit reached ({elapsed:.0f}s). Stopping.")
                        stop_reason = "wall_time"
                        break

                t0 = time.monotonic()

                td_device = td.to(self.device)
                if self.total_frames < self.config.warmup_steps:
                    action = torch.rand(self.env.action_spec.shape) * 2 - 1
                    td_device["action"] = action.to(self.device)
                else:
                    with torch.no_grad():
                        amp_ctx = _amp_context(self.config.use_amp, self.device)
                        with amp_ctx:
                            td_device = self.actor(td_device)

                td_env = td_device.to(td.device)
                if is_vec:
                    next_td, td_reset = self.env.step_and_maybe_reset(td_env)
                else:
                    next_td = self.env.step(td_env)
                    td_reset = None
                env_dt = time.monotonic() - t0

                t0 = time.monotonic()
                store_td = next_td.clone()
                if "next" in store_td.keys():
                    nxt = store_td["next"]
                    if "reward" in nxt.keys():
                        store_td["reward"] = nxt["reward"]
                    if "done" in nxt.keys():
                        store_td["done"] = nxt["done"]
                if is_vec:
                    self.replay_buffer.extend(store_td)
                else:
                    self.replay_buffer.add(store_td)
                data_dt = time.monotonic() - t0

                frames_this_step = num_envs
                self.total_frames += frames_this_step
                pbar.update(frames_this_step)

                step_result = next_td["next"] if "next" in next_td.keys() else next_td

                if is_vec:
                    rewards = step_result["reward"].cpu().numpy().reshape(num_envs)
                    dones = step_result["done"].cpu().numpy().reshape(num_envs)
                    episode_rewards += rewards
                    episode_lengths += 1

                    done_mask = dones.astype(bool)
                    if done_mask.any():
                        for i in np.where(done_mask)[0]:
                            self.total_episodes += 1
                            metrics = {
                                "episode_reward": float(episode_rewards[i]),
                                "episode_length": int(episode_lengths[i]),
                                "total_frames": self.total_frames,
                            }
                            all_metrics.append(metrics)

                            if episode_rewards[i] > self.best_reward:
                                self.best_reward = float(episode_rewards[i])
                                self.save_checkpoint("best")

                            if self.total_episodes % self.config.log_interval == 0:
                                tqdm.write(
                                    f"Episode {self.total_episodes}: "
                                    f"reward={episode_rewards[i]:.2f}, "
                                    f"length={episode_lengths[i]}"
                                )

                            self._log_episode_metrics(episode_rewards[i], episode_lengths[i])

                            if callback:
                                callback(metrics)

                        episode_rewards[done_mask] = 0.0
                        episode_lengths[done_mask] = 0

                    td = td_reset
                else:
                    episode_rewards[0] += step_result["reward"].item()
                    episode_lengths[0] += 1

                    if step_result["done"].item():
                        self.total_episodes += 1
                        metrics = {
                            "episode_reward": float(episode_rewards[0]),
                            "episode_length": int(episode_lengths[0]),
                            "total_frames": self.total_frames,
                        }
                        all_metrics.append(metrics)

                        if episode_rewards[0] > self.best_reward:
                            self.best_reward = float(episode_rewards[0])
                            self.save_checkpoint("best")

                        if self.total_episodes % self.config.log_interval == 0:
                            tqdm.write(
                                f"Episode {self.total_episodes}: "
                                f"reward={episode_rewards[0]:.2f}, "
                                f"length={episode_lengths[0]}"
                            )

                        self._log_episode_metrics(episode_rewards[0], episode_lengths[0])

                        td = self.env.reset()
                        episode_rewards[0] = 0.0
                        episode_lengths[0] = 0

                        if callback:
                            callback(metrics)
                    else:
                        td = step_result

                t0 = time.monotonic()
                update_metrics = {}
                if (
                    self.total_frames >= self.config.warmup_steps
                    and self.total_frames % self.config.update_frequency == 0
                ):
                    for _ in range(self.config.num_updates):
                        update_metrics = self._update()
                backward_dt = time.monotonic() - t0

                t0 = time.monotonic()
                if update_metrics:
                    self._log_train_metrics(update_metrics)
                if self.total_frames % (self.config.save_interval * 1000) == 0:
                    self.save_checkpoint(f"step_{self.total_frames}")
                overhead_dt = time.monotonic() - t0

                total_dt = env_dt + data_dt + backward_dt + overhead_dt
                if update_metrics:
                    update_metrics["timing/env_step_seconds"] = env_dt
                    update_metrics["timing/backward_seconds"] = backward_dt
                    self._write_metrics_jsonl(update_metrics)

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

    def _update(self) -> Dict[str, float]:
        """Perform single SAC update."""
        amp_ctx = _amp_context(self.config.use_amp, self.device)

        batch = self.replay_buffer.sample()
        obs = batch["observation"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next"]["observation"]
        done = batch["done"].float()

        with torch.no_grad():
            with amp_ctx:
                next_td = TensorDict({"observation": next_obs}, batch_size=obs.shape[0])
                next_td = self.actor(next_td)
                next_action = next_td["action"]
                next_log_prob = next_td["action_log_prob"]

                q1_target, q2_target = self.critic_target(next_obs, next_action)
                q_target = torch.min(q1_target, q2_target)
                if self._use_entropy:
                    q_target = q_target - self.alpha * next_log_prob.unsqueeze(-1)
                target_value = reward + (1 - done) * self.config.gamma * q_target

        with amp_ctx:
            q1, q2 = self.critic(obs, action)
            critic_loss = nn.functional.mse_loss(q1, target_value) + nn.functional.mse_loss(
                q2, target_value
            )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = compute_grad_norm(self.critic)
        if self.config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()

        actor_loss = None
        alpha_loss = None
        actor_grad_norm = None
        self._update_count += 1
        if self._update_count % self.config.actor_update_frequency == 0:
            with amp_ctx:
                td = TensorDict({"observation": obs}, batch_size=obs.shape[0])
                td = self.actor(td)
                new_action = td["action"]
                log_prob = td["action_log_prob"]

                q1_new, q2_new = self.critic(obs, new_action)
                q_new = torch.min(q1_new, q2_new)

                if self._use_entropy:
                    actor_loss = (self.alpha * log_prob.unsqueeze(-1) - q_new).mean()
                else:
                    actor_loss = (-q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = compute_grad_norm(self.actor)
            actor_clip = self.config.actor_max_grad_norm or self.config.max_grad_norm
            if actor_clip is not None:
                nn.utils.clip_grad_norm_(self.actor.parameters(), actor_clip)
            self.actor_optimizer.step()

            if self.config.auto_alpha:
                with amp_ctx:
                    alpha_loss = -(
                        self.log_alpha * (log_prob + self.target_entropy).detach()
                    ).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

        if self._update_count % self.config.soft_update_period == 0:
            self._soft_update()

        metrics = {
            "critic_loss": critic_loss.item(),
            "alpha": self.alpha.item(),
            "critic_grad_norm": critic_grad_norm,
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
        }

        if actor_loss is not None:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_grad_norm"] = actor_grad_norm
        if alpha_loss is not None:
            metrics["alpha_loss"] = alpha_loss.item()
        return metrics

    def _soft_update(self) -> None:
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )

    def _log_episode_metrics(self, episode_reward: float, episode_length: int) -> None:
        wandb_utils.log_metrics(
            self.wandb_run,
            {
                "episode/mean_reward": episode_reward,
                "episode/mean_length": float(episode_length),
                "episode/count": self.total_episodes,
                "tracking/best_reward": self.best_reward,
            },
            step=self.total_frames,
        )

    def _log_train_metrics(self, metrics: Dict[str, float]) -> None:
        wandb_log = {}
        for key in ("critic_loss", "actor_loss", "alpha", "alpha_loss"):
            if key in metrics:
                wandb_log[f"train/{key}"] = metrics[key]
        for key in ("q1_mean", "q2_mean"):
            if key in metrics:
                wandb_log[f"q_values/{key}"] = metrics[key]
        for key in ("critic_grad_norm", "actor_grad_norm"):
            if key in metrics:
                wandb_log[f"gradients/{key}"] = metrics[key]
        wall = time.monotonic() - self._train_start_time
        wandb_log["timing/wall_clock_mins"] = wall / 60.0

        for key in ("timing/env_step_seconds", "timing/backward_seconds"):
            if key in metrics:
                wandb_log[key] = metrics[key]

        sys_metrics = collect_system_metrics(self.device, self._system_log_counter, 10)
        wandb_log.update(sys_metrics)

        wandb_utils.log_metrics(self.wandb_run, wandb_log, step=self.total_frames)

    def save_checkpoint(self, name: str) -> None:
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha,
            "total_frames": self.total_frames,
            "total_episodes": self.total_episodes,
            "best_reward": self.best_reward,
            "config": self.config,
        }

        if self.config.auto_alpha:
            checkpoint["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()

        path = self.save_dir / f"{name}.pt"
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.log_alpha = checkpoint["log_alpha"]
        self.total_frames = checkpoint["total_frames"]
        self.total_episodes = checkpoint["total_episodes"]
        self.best_reward = checkpoint["best_reward"]

        if self.config.auto_alpha and "alpha_optimizer_state_dict" in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
