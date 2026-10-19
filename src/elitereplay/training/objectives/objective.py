"""Source-wise reductions, invariant to DP and dynamic microbatch boundaries."""
from __future__ import annotations

import torch
from . import grpo


def loss(logprob, old_logprob, action_mask, advantages, expert_mask,
         online_token_count, auxiliary_token_count, *, method, dp_size=1,
         sft_weight=0.05, clip_low=0.2, clip_high=0.28,
         sft_reduction="token_mean", auxiliary_response_count=None):
    if method != "EliteReplay":
        raise ValueError("Only the EliteReplay objective is supported")
    expert = expert_mask.bool().reshape(-1)
    online = ~expert
    # A microbatch can contain only one source; retain a differentiable zero.
    rl = logprob.sum() * 0
    aux = logprob.sum() * 0
    if online.any():
        terms = grpo.token_loss(logprob[online], old_logprob[online], advantages[online],
                                clip_low, clip_high)
        rl = (terms * action_mask[online]).sum() / online_token_count.reshape(-1)[0]
    if expert.any():
        terms = -logprob[expert]
        if sft_reduction == "response_mean":
            if auxiliary_response_count is None:
                raise ValueError("Response-mean SFT requires the global auxiliary response count")
            lengths = action_mask[expert].sum(-1)
            if (lengths <= 0).any():
                raise ValueError("A replay response has no supervised tokens")
            aux = ((terms * action_mask[expert]).sum(-1) / lengths).sum() / auxiliary_response_count.reshape(-1)[0]
        else:
            if sft_reduction not in ("token_mean", "response_mean"):
                raise ValueError("Unknown SFT reduction")
            aux = (terms * action_mask[expert]).sum() / auxiliary_token_count.reshape(-1)[0]
    rl_weight = 1 - sft_weight
    aux_weight = sft_weight
    # FSDP averages rank gradients; microbatch gradients are summed by VERL.
    total = dp_size * (rl_weight * rl + aux_weight * aux)
    return total, {"baseline/rl_loss_contribution": float(rl.detach()),
                   "baseline/aux_loss_contribution": float(aux.detach())}
