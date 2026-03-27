"""Self-contained soft manipulator follow-target (Choi & Tong, 2025).

A standalone package for training SAC on the follow-target task using a
clamped soft manipulator with DisMech physics (or mock backend fallback).
"""

from .env import SoftManipulatorEnv

__all__ = ["SoftManipulatorEnv"]
