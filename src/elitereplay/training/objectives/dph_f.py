"""DPH-F as released: forward KL with the default low_var_kl estimator.

The upstream actor passes stored reference log probabilities first to
kl_penalty(..., kl_penalty='low_var_kl'). Preserve both clamps and gradients.
Source: seamoke/DPH-RL, dp_actor.py:508 and core_algos.py:kl_penalty.
This finite-sample estimator is not interchangeable with plain SFT loss.
"""
import torch


def token_loss(logprob, reference_logprob):
    log_ratio = (logprob - reference_logprob.detach()).clamp(min=-20, max=20)
    return (log_ratio.exp() - log_ratio - 1).contiguous().clamp(min=-10, max=10)
