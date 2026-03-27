"""Actor and critic networks for SAC and PPO (self-contained).

Provides:
    ActorNetwork     — MLP that outputs (mean, std) for Gaussian policy
    CriticNetwork    — MLP for state-value V(s) (PPO)
    QNetwork         — Q(s, a) network (SAC)
    TwinQNetwork     — Twin Q-networks for min-Q trick (SAC)
    create_actor()   — Build TorchRL ProbabilisticActor
    create_critic()  — Build TorchRL ValueOperator (PPO)
"""

from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from torchrl.modules import (
    ProbabilisticActor,
    TanhNormal,
    NormalParamExtractor,
    ValueOperator,
)
try:
    from torchrl.data import BoundedTensorSpec, CompositeSpec
except ImportError:
    from torchrl.data import Bounded as BoundedTensorSpec, Composite as CompositeSpec
from tensordict.nn import TensorDictModule, TensorDictSequential

from choi2025_follow_target.config import ActorConfig, CriticConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name.lower()]()


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------


class ActorNetwork(nn.Module):
    """MLP network for actor (policy). Outputs mean and std for Gaussian policy."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str = "tanh",
        min_std: float = 0.1,
        max_std: float = 1.0,
        init_std: float = 0.5,
        ortho_init: bool = True,
        init_gain: float = 0.01,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.min_std = min_std
        self.max_std = max_std

        layers = []
        prev_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(get_activation(activation))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)

        init_log_std = torch.log(torch.tensor(init_std))
        nn.init.constant_(self.log_std_head.bias, init_log_std.item())

        if ortho_init:
            self._apply_ortho_init(init_gain)

    def _apply_ortho_init(self, final_gain: float) -> None:
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

        nn.init.orthogonal_(self.mean_head.weight, gain=final_gain)
        nn.init.constant_(self.mean_head.bias, 0.0)
        nn.init.orthogonal_(self.log_std_head.weight, gain=final_gain)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.mlp(obs)
        mean = self.mean_head(features)
        mean = torch.clamp(mean, -5.0, 5.0)

        log_std = self.log_std_head(features)
        log_std = torch.clamp(
            log_std,
            min=torch.log(torch.tensor(self.min_std, device=log_std.device)),
            max=torch.log(torch.tensor(self.max_std, device=log_std.device)),
        )
        std = torch.exp(log_std)

        return mean, std


# ---------------------------------------------------------------------------
# State-value critic (PPO)
# ---------------------------------------------------------------------------


class CriticNetwork(nn.Module):
    """MLP network for state-value function V(s)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_outputs: int = 1,
        activation: str = "tanh",
        ortho_init: bool = True,
        init_gain: float = 1.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(get_activation(activation))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_dim, num_outputs)

        if ortho_init:
            self._apply_ortho_init(init_gain)

    def _apply_ortho_init(self, final_gain: float) -> None:
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

        nn.init.orthogonal_(self.value_head.weight, gain=final_gain)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.mlp(x)
        return self.value_head(features)


# ---------------------------------------------------------------------------
# Q-value critic (SAC)
# ---------------------------------------------------------------------------


class QNetwork(nn.Module):
    """Q-network that takes observation and action as separate inputs."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        ortho_init: bool = True,
        init_gain: float = 1.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        obs_layers = []
        prev_dim = obs_dim
        first_hidden = hidden_dims[0] if hidden_dims else 256

        obs_layers.append(nn.Linear(prev_dim, first_hidden))
        if use_layer_norm:
            obs_layers.append(nn.LayerNorm(first_hidden))
        obs_layers.append(get_activation(activation))

        self.obs_encoder = nn.Sequential(*obs_layers)

        combined_layers = []
        prev_dim = first_hidden + action_dim

        for hidden_dim in hidden_dims[1:]:
            combined_layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                combined_layers.append(nn.LayerNorm(hidden_dim))
            combined_layers.append(get_activation(activation))
            prev_dim = hidden_dim

        self.combined = nn.Sequential(*combined_layers)
        self.q_head = nn.Linear(prev_dim, 1)

        if ortho_init:
            self._apply_ortho_init(init_gain)

    def _apply_ortho_init(self, final_gain: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

        nn.init.orthogonal_(self.q_head.weight, gain=final_gain)
        nn.init.constant_(self.q_head.bias, 0.0)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs_features = self.obs_encoder(obs)
        combined = torch.cat([obs_features, action], dim=-1)
        features = self.combined(combined)
        q_value = self.q_head(features)
        return q_value


class TwinQNetwork(nn.Module):
    """Twin Q-networks for SAC (min of two Q-functions)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        ortho_init: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.q1 = QNetwork(
            obs_dim, action_dim, hidden_dims, activation, ortho_init, 1.0, use_layer_norm
        )
        self.q2 = QNetwork(
            obs_dim, action_dim, hidden_dims, activation, ortho_init, 1.0, use_layer_norm
        )

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q1 = self.q1(obs, action)
        q2 = self.q2(obs, action)
        return q1, q2

    def min_q(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_actor(
    obs_dim: int,
    action_spec: BoundedTensorSpec,
    config: Optional[ActorConfig] = None,
    device: str = "cpu",
) -> ProbabilisticActor:
    """Create TorchRL ProbabilisticActor."""
    config = config or ActorConfig()
    action_dim = action_spec.shape[-1]

    net = ActorNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=config.hidden_dims,
        activation=config.activation,
        min_std=config.min_std,
        max_std=config.max_std,
        init_std=config.init_std,
        ortho_init=config.ortho_init,
        init_gain=config.init_gain,
        use_layer_norm=config.use_layer_norm,
    ).to(device)

    actor_module = TensorDictModule(
        net,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )

    actor = ProbabilisticActor(
        module=actor_module,
        spec=action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
        },
        return_log_prob=True,
        default_interaction_type="random",
    )

    return actor


def create_critic(
    obs_dim: int,
    config: Optional[CriticConfig] = None,
    device: str = "cpu",
) -> ValueOperator:
    """Create TorchRL ValueOperator for state-value function V(s) (PPO)."""
    config = config or CriticConfig()

    net = CriticNetwork(
        input_dim=obs_dim,
        hidden_dims=config.hidden_dims,
        num_outputs=1,
        activation=config.activation,
        ortho_init=config.ortho_init,
        init_gain=config.init_gain,
        use_layer_norm=config.use_layer_norm,
    ).to(device)

    return ValueOperator(module=net, in_keys=["observation"])
