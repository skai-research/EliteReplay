"""MaxRL's centered binary estimator; do not standardize this again.

Reference: https://arxiv.org/abs/2602.02710 and
https://github.com/tajwarfahim/maxrl/blob/main/verl/trainer/ppo/core_algos.py
This is an implementation of the estimator, not a TailRL reward transform.
"""
from __future__ import annotations

import torch
from .grpo import check_rewards


def advantages(rewards: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    rewards = check_rewards(rewards)
    mean = rewards.mean()
    return (rewards - mean) / (mean + epsilon)
