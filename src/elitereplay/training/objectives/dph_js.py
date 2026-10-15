"""DPH-JS (Generator), using the released code's coefficient convention.

The authors' js_low_var function implements 2*JS (omits the paper's 1/2).
We preserve that convention so eta=0.2 has the released-code meaning.
Reference: seamoke/DPH-RL, verl/trainer/ppo/core_algos.py:js_low_var;
paper https://arxiv.org/abs/2509.07430, sections 4.2 and 6.2.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from .grpo import check_rewards


def is_preserved(rewards, threshold=8):
    values = check_rewards(torch.as_tensor(rewards))
    if not 1 <= threshold <= len(values):
        raise ValueError("Invalid DPH preservation threshold")
    return int(values.sum()) >= threshold


def informative_group(rewards):
    values = check_rewards(torch.as_tensor(rewards))
    return 0 < int(values.sum()) < len(values)


def token_loss(logprob, reference_logprob):
    """Stable generator on stored reference-policy actions, with live gradients.

    No detach of the ratio: that would change the f-divergence gradient.
    Reference log probabilities must be from the frozen initialization, not
    from the previous online policy or from a retokenized canonical answer.
    """
    log_ratio = logprob - reference_logprob.detach()
    ratio = log_ratio.exp()
    # Algebraically u*log(u) - (u+1)*log((u+1)/2), evaluated without
    # subtracting two quantities proportional to u*log(u) for large u.
    result = ratio * (math.log(2) - F.softplus(-log_ratio)) - F.softplus(log_ratio) + math.log(2)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Non-finite DPH generator; inspect reference log probabilities")
    return result
