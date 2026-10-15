# Modified for EliteReplay; see vendor/runtime/README.md for provenance.
# -*- coding: utf-8 -*-
"""Trinity policy loss function for veRL's engine-based training API.
This module provides a loss function compatible with veRL's
BaseEngine.forward_backward_batch() interface, replacing the old
DataParallelPPOActor.update_policy() approach.
The loss function signature expected by veRL's engine:
    def loss_fn(model_output, data: TensorDict, dp_group=None) -> (loss, metrics)
"""
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import no_padding_2_padding

from trinity.algorithm import ENTROPY_LOSS_FN, KL_FN, POLICY_LOSS_FN
from trinity.algorithm.kl_fn.kl_fn import DummyKLFn
from trinity.algorithm.policy_loss_fn.chord_policy_loss import MIXCHORDPolicyLossFn
from trinity.algorithm.policy_loss_fn.mix_policy_loss import MIXPolicyLossFn
from trinity.algorithm.utils import prefix_metrics
from trinity.common.config import AlgorithmConfig


def _mix_microbatch_loss_scale(
    *,
    micro_batch_size: int,
    global_batch_size: int,
    dp_size: int,
    expected_local_batch_size: int,
) -> float:
    """Restore the outer scale used by the legacy MIX actor.

    MIX and MIX-CHORD normalize each source inside the policy loss using an
    ``experience_per_gpu / micro_batch_size`` factor.  The legacy actor then
    multiplied the combined loss by ``micro_batch_size / experience_per_gpu``
    before backward.  The engine-based adapter must do the same; otherwise
    dynamic micro-batches receive length-dependent gradient weights.
    """

    if micro_batch_size <= 0 or global_batch_size <= 0 or dp_size <= 0:
        raise RuntimeError(
            "MIX loss scaling requires positive micro/global batch sizes and dp_size"
        )
    if global_batch_size % dp_size != 0:
        raise RuntimeError(
            f"global batch {global_batch_size} is not divisible by dp_size {dp_size}"
        )
    local_batch_size = global_batch_size // dp_size
    if local_batch_size != expected_local_batch_size:
        raise RuntimeError(
            "MIX loss batch contract changed: "
            f"engine local batch={local_batch_size}, policy expected "
            f"{expected_local_batch_size}"
        )
    if micro_batch_size > local_batch_size:
        raise RuntimeError(
            f"micro-batch {micro_batch_size} exceeds local batch {local_batch_size}"
        )
    return micro_batch_size / local_batch_size


class TrinityPolicyLoss:
    """Picklable policy loss callable for veRL's engine API.
    Wraps Trinity's POLICY_LOSS_FN, KL_FN, and ENTROPY_LOSS_FN registries
    into a single callable that can be serialized by Ray and sent to remote
    workers via set_loss_fn().
    """

    def __init__(self, algo_config: AlgorithmConfig):
        self.policy_loss_fn = POLICY_LOSS_FN.get(algo_config.policy_loss_fn)(
            backend="verl", **algo_config.policy_loss_fn_args
        )
        self.kl_loss_fn = KL_FN.get(algo_config.kl_loss_fn)(**algo_config.kl_loss_fn_args)
        self.entropy_loss_fn = ENTROPY_LOSS_FN.get(algo_config.entropy_loss_fn)(
            **algo_config.entropy_loss_fn_args
        )
        self.calculate_entropy = algo_config.entropy_loss_fn != "none"
        self.loss_agg_mode = algo_config.loss_agg_mode
        self.use_kl_loss = not isinstance(self.kl_loss_fn, DummyKLFn)
        self._restore_mix_microbatch_scale = isinstance(
            self.policy_loss_fn, (MIXPolicyLossFn, MIXCHORDPolicyLossFn)
        )

    def __call__(
        self,
        model_output: dict,
        data: TensorDict,
        dp_group=None,
    ) -> tuple[torch.Tensor, dict]:
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        entropy = model_output.get("entropy", None)
        if entropy is not None:
            entropy = no_padding_2_padding(entropy, data)

        fields = ["response_mask"]
        fields.extend(self.policy_loss_fn.select_keys)
        if self.use_kl_loss:
            fields.append("ref_log_prob")
            if "old_log_probs" in data.keys():
                fields.append("old_log_probs")
        fields = list(dict.fromkeys(fields))
        padded_data = data.select(*fields).to_padded_tensor()

        response_mask = padded_data["response_mask"].to(bool)
        model_inputs = {"response_mask": response_mask}
        for key in fields:
            if key == "response_mask":
                continue
            if key in padded_data.keys():
                model_inputs[key] = padded_data[key]

        metrics = {}

        pg_loss, pg_loss_metrics = self.policy_loss_fn(logprob=log_prob, **model_inputs)
        prefix_metrics(src_metrics=pg_loss_metrics, prefix="actor", dst_metrics=metrics)
        policy_loss = pg_loss

        if self.calculate_entropy and entropy is not None:
            entropy_loss, entropy_loss_metrics = self.entropy_loss_fn(
                entropy=entropy,
                action_mask=response_mask,
                loss_agg_mode=self.loss_agg_mode,
                **model_inputs,
            )
            prefix_metrics(src_metrics=entropy_loss_metrics, prefix="actor", dst_metrics=metrics)
            policy_loss = policy_loss - entropy_loss

        if self.use_kl_loss:
            kl_loss, kl_loss_metrics = self.kl_loss_fn.calculate_kl_loss(
                logprob=log_prob,
                ref_logprob=model_inputs.get("ref_log_prob", None),
                response_mask=response_mask,
                loss_agg_mode=self.loss_agg_mode,
                old_logprob=model_inputs.get("old_log_probs", None),
            )
            prefix_metrics(src_metrics=kl_loss_metrics, prefix="actor", dst_metrics=metrics)
            policy_loss = policy_loss + kl_loss

        if self._restore_mix_microbatch_scale:
            global_batch_size = int(
                tu.get_non_tensor_data(
                    data=data, key="global_batch_size", default=0
                )
            )
            dp_size = int(
                tu.get_non_tensor_data(data=data, key="dp_size", default=0)
            )
            loss_scale = _mix_microbatch_loss_scale(
                micro_batch_size=int(log_prob.shape[0]),
                global_batch_size=global_batch_size,
                dp_size=dp_size,
                expected_local_batch_size=int(self.policy_loss_fn.experience_per_gpu),
            )
            policy_loss = policy_loss * loss_scale
            metrics["micro_batch_loss_scale"] = loss_scale

        # veRL sums these scaled micro-batch losses and then averages gradients
        # across DP ranks, matching the legacy actor's normalization.
        metrics["final_loss"] = policy_loss.detach().item()

        return policy_loss, metrics

    def __repr__(self) -> str:
        return (
            f"TrinityPolicyLoss(policy={self.policy_loss_fn.__class__.__name__}, "
            f"kl={self.kl_loss_fn.__class__.__name__}, "
            f"entropy={self.entropy_loss_fn.__class__.__name__})"
        )


def build_trinity_loss(algo_config: AlgorithmConfig) -> TrinityPolicyLoss:
    """Build a TrinityPolicyLoss instance for veRL's engine API."""
    return TrinityPolicyLoss(algo_config)
