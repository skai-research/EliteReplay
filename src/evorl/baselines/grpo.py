"""Binary-reward GRPO and the shared, asymmetric PPO clipping objective.

Only response tokens contribute. Group standard deviation uses correction=1,
matching the VERL reference. Constant-reward groups have zero advantages.
"""
from __future__ import annotations

import torch


def check_rewards(rewards: torch.Tensor) -> torch.Tensor:
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("A rollout group must contain at least two rewards")
    if not torch.isfinite(rewards).all() or not ((rewards == 0) | (rewards == 1)).all():
        raise ValueError("All baselines require binary, finite correctness rewards")
    return rewards


def advantages(rewards: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    rewards = check_rewards(rewards)
    return (rewards - rewards.mean()) / (rewards.std(unbiased=True) + epsilon)


def token_loss(logprob, old_logprob, advantage, clip_low=0.2, clip_high=0.28):
    """PPO-clip without an additional dual-clip or advantage normalization."""
    ratio = (logprob - old_logprob.detach()).clamp(-20, 20).exp()
    clipped = ratio.clamp(1 - clip_low, 1 + clip_high)
    return torch.maximum(-advantage * ratio, -advantage * clipped)
