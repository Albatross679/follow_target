# Choi2025 Follow-Target: Self-Contained SAC Training

Standalone package for training SAC on the follow-target soft manipulator task
from **Choi & Tong (2025)**, *Rapidly Learning Soft Robot Control via Implicit Time-Stepping*.

## Quick Start

```bash
# Install (from this directory)
pip install -e .

# Train (mock physics, 32 parallel envs)
python -m choi2025_follow_target.train --num-envs 32 --total-frames 1000000

# Train with wall-clock limit
python -m choi2025_follow_target.train --max-wall-time 30m

# Record video from checkpoint
python -m choi2025_follow_target.record \
    --checkpoint output/*/checkpoints/best.pt \
    --output media/follow_target.mp4

# Record passive dynamics (no checkpoint)
python -m choi2025_follow_target.record --steps 200
```

## Structure

| File | Role |
|------|------|
| `config.py` | All configuration dataclasses (physics, env, network, SAC) |
| `env.py` | TorchRL environment (DisMech or mock physics) |
| `train.py` | Training entry point |
| `record.py` | Video recording from checkpoint |
| `control.py` | Delta curvature controller with Voronoi smoothing |
| `rewards.py` | Reward functions |
| `tasks.py` | Target generation and obstacle management |
| `networks.py` | Actor and critic neural networks |
| `trainer.py` | SAC trainer loop |
| `wandb_utils.py` | Weights & Biases integration |

## Dependencies

- PyTorch >= 2.0
- TorchRL >= 0.3
- NumPy, Matplotlib, tqdm, wandb
- **Optional**: DisMech (for real physics; mock backend used as fallback)
