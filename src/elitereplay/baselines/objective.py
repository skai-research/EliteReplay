"""Source-wise reductions, invariant to DP and dynamic microbatch boundaries."""
from __future__ import annotations

import torch
from . import grpo, dph_f, dph_js


def loss(logprob, old_logprob, action_mask, advantages, expert_mask,
         online_token_count, auxiliary_token_count, *, method, dp_size=1,
         sft_weight=0.05, js_weight=0.2, forward_weight=0.01, clip_low=0.2, clip_high=0.28,
         sft_reduction="token_mean", auxiliary_response_count=None):
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
        if method == "dph_f":
            terms = dph_f.token_loss(logprob[expert], old_logprob[expert])
        elif method == "dph_js":
            terms = dph_js.token_loss(logprob[expert], old_logprob[expert])
        elif method in ("IID_Replay", "EliteReplay"):
            terms = -logprob[expert]
        else:
            raise ValueError(f"Unexpected replay in {method}")
        if method in ("IID_Replay", "EliteReplay") and sft_reduction == "response_mean":
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
    rl_weight = 1 - sft_weight if method in ("IID_Replay", "EliteReplay") else 1
    aux_weight = forward_weight if method == "dph_f" else js_weight if method == "dph_js" else sft_weight
    # FSDP averages rank gradients; microbatch gradients are summed by VERL.
    total = dp_size * (rl_weight * rl + aux_weight * aux)
    return total, {"baseline/rl_loss_contribution": float(rl.detach()),
                   "baseline/aux_loss_contribution": float(aux.detach())}
