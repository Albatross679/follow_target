"""Shared Weights & Biases integration (self-contained)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def setup_run(config, run_dir: Path, *, resume_id: Optional[str] = None) -> Any:
    """Initialize a W&B run with proper metric axes."""
    if not getattr(config, "wandb", None) or not config.wandb.enabled:
        return None

    import wandb

    wandb_cfg = config.wandb
    init_kwargs = dict(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity or None,
        group=wandb_cfg.group or None,
        tags=wandb_cfg.tags or None,
        name=config.name,
        config=asdict(config),
        dir=str(run_dir),
    )
    if resume_id:
        init_kwargs["resume"] = "allow"
        init_kwargs["id"] = resume_id

    run = wandb.init(**init_kwargs)

    wandb.define_metric("train/*", step_metric="total_frames")
    wandb.define_metric("episode/*", step_metric="total_frames")
    wandb.define_metric("timing/*", step_metric="total_frames")
    wandb.define_metric("system/*", step_metric="total_frames")
    wandb.define_metric("tracking/*", step_metric="total_frames")
    wandb.define_metric("gradients/*", step_metric="total_frames")
    wandb.define_metric("diagnostics/*", step_metric="total_frames")
    wandb.define_metric("q_values/*", step_metric="total_frames")

    return run


def log_metrics(run, metrics: Dict[str, Any], step: int) -> None:
    if run is None:
        return
    metrics["total_frames"] = step
    run.log(metrics, step=step)


def log_extra_params(run, params: Dict[str, Any]) -> None:
    if run is None:
        return
    run.config.update(params, allow_val_change=True)
    for k, v in params.items():
        run.summary[k] = v


def log_model_artifact(
    run, checkpoint_path: Path, artifact_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    if run is None:
        return
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return

    import wandb

    artifact = wandb.Artifact(name=artifact_name, type="model", metadata=metadata or {})
    artifact.add_file(str(checkpoint_path))
    run.log_artifact(artifact)


def end_run(run) -> None:
    if run is None:
        return
    run.finish()


def _count_parameters(module: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def collect_hardware_info(device: str) -> Dict[str, str]:
    info: Dict[str, str] = {}
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        dev = torch.device(device)
        info["gpu_name"] = torch.cuda.get_device_name(dev)
        info["gpu_memory_total_gb"] = f"{torch.cuda.get_device_properties(dev).total_memory / 1e9:.1f}"
    try:
        import psutil
        info["cpu_count"] = str(psutil.cpu_count(logical=True))
        info["ram_total_gb"] = f"{psutil.virtual_memory().total / 1e9:.1f}"
    except ImportError:
        pass
    return info
