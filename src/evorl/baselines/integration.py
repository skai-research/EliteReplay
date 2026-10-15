"""Trinity adapter. Algorithm formulas live in the adjacent small modules."""
from __future__ import annotations

import copy
import importlib.util
import json
import logging
import random
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import torch
from trinity.algorithm.policy_loss_fn import POLICY_LOSS_FN
from trinity.algorithm.policy_loss_fn.policy_loss_fn import PolicyLossFn
from trinity.algorithm.sample_strategy import SAMPLE_STRATEGY
from trinity.algorithm.sample_strategy.sample_strategy import SampleStrategy
from trinity.algorithm.sample_strategy.utils import representative_sample
from trinity.buffer import get_buffer_reader
from trinity.common.experience import CustomField, EID, Experience
from trinity.common.workflows import WORKFLOWS

from . import grpo, maxrl, setpo, objective
from .batching import arrange, compact_zero_loss
from .config import DPH_METHODS, ROOT
from .replay import Replay
from evorl.EliteReplay.trinity_support import ON_POLICY_SOURCE, render_sft_with_terminal_eos

# Register only this suite's entry plugin. Load the existing verified adapters
# once under a distinct module name; don't scan historical baseline plugins.
_spec = importlib.util.spec_from_file_location("evorl_baseline_domain_support", ROOT / "plugins/EliteReplay_trinity.py")
support = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = support
_spec.loader.exec_module(support)


@lru_cache(maxsize=4)
def exploration_rows(filename):
    import pyarrow.parquet as pq
    return pq.read_table(filename).to_pylist()


@WORKFLOWS.register_module("evorl_binary_workflow")
class BinaryWorkflow(support.EliteReplayWorkflow):
    def _score_response(self, response, index):
        super()._score_response(response, index)
        response.info["partial_score"] = response.reward
        response.reward = float(response.info["fully_correct"] and response.info["strict_parse"])
        response.metrics["reward/verification"] = response.reward
        response.info["problem_id"] = self.raw_task["extra_info"]["problem_id"]

    def run(self):
        dynamic = self.workflow_args.get("dynamic_sampling", False)
        attempts = self.workflow_args.get("max_resample_groups", 64) if dynamic else 1
        generated = tokens = 0
        rng = None
        # Continue sampling fresh questions after a rejected block instead of
        # throwing away the task and leaving the synchronous batch one group
        # short. Sixteen blocks bound pathological no-signal runs; the rollout
        # timeout remains an independent limit. Dynamic mode rejects homogeneous
        # groups; fixed-batch mode retains them with zero GRPO advantages.
        all_wrong = all_correct = 0
        for attempt in range(attempts * (16 if dynamic else 1)):
            responses = super().run()
            generated += len(responses)
            tokens += sum(len(e.tokens) - e.prompt_length for e in responses)
            if rng is None:
                # Policy version changes the resampling stream each iteration.
                rng = random.Random(f"{self.workflow_args.get('seed', 42)}:dph:"
                                    f"{self.raw_task['extra_info']['problem_id']}:"
                                    f"{responses[0].info.get('model_version')}")
            successes = sum(e.reward for e in responses)
            if not dynamic or 0 < successes < len(responses):
                # Store group totals on only one response to avoid G-fold counts.
                responses[0].info.update(generated_responses=generated, generated_tokens=tokens,
                                         discarded_groups=attempt,
                                         discarded_all_wrong_groups=all_wrong,
                                         discarded_all_correct_groups=all_correct,
                                         replenishment_blocks=attempt // attempts)
                return responses
            all_wrong += int(successes == 0)
            all_correct += int(successes == len(responses))
            if (attempt + 1) % attempts == 0:
                logging.getLogger(__name__).warning(
                    "DPH replenishing after %d rejected groups (%d all-wrong, %d all-correct)",
                    attempt + 1, all_wrong, all_correct)
            rows = exploration_rows(self.workflow_args["exploration_path"])
            self.raw_task = rng.choice(rows)
        raise RuntimeError(f"DPH exhausted {attempts * 16} groups without mixed rewards "
                           f"({all_wrong} all-wrong, {all_correct} all-correct)")


def fresh_groups(experiences, step, count, repeat):
    if len(experiences) != count:
        raise RuntimeError(f"Expected {count} fresh responses, received {len(experiences)}")
    groups = defaultdict(list)
    for exp in experiences:
        info = exp.info or {}
        if info.get("sample_source") != ON_POLICY_SOURCE or info.get("is_expert"):
            raise RuntimeError("Replay entered the on-policy stream")
        if info.get("model_version") != step - 1 or info.get("model_version_drift", 0) != 0:
            raise RuntimeError("Stale or mixed-version online rollout")
        groups[exp.eid.tid].append(exp)
    for group in groups.values():
        if len(group) != repeat or len({e.eid.run for e in group}) != repeat:
            raise RuntimeError("Incomplete/duplicated rollout group")
        if len({tuple(e.tokens[:e.prompt_length].tolist()) for e in group}) != 1:
            raise RuntimeError("A rollout group contains different prompts")
    return list(groups.values())


@SAMPLE_STRATEGY.register_module("evorl_baselines")
class BaselineSampler(SampleStrategy):
    def __init__(self, buffer_config, **kwargs):
        super().__init__(buffer_config)
        self.args = kwargs
        self.method = kwargs["method"]
        storage = copy.deepcopy(buffer_config.trainer_input.experience_buffer)
        storage.batch_size = kwargs["online_count"]
        self.reader = get_buffer_reader(storage)
        self.replay = None
        self.tokenizer = None
        self.embedder = None
        if kwargs["replay_count"] or self.method == "setpo":
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(kwargs["policy_model_path"], local_files_only=True)
        if kwargs["replay_count"]:
            rows = [json.loads(line) for line in Path(kwargs["replay_path"]).read_text().splitlines() if line.strip()]
            kind = {"EliteReplay": "qd", "IID_Replay": "iid", "dph_js": "dph_js", "dph_f": "dph_f"}[self.method]
            self.replay = Replay(rows, method=kind, seed=kwargs["seed"],
                                 sampling=kwargs.get("replay_sampling", "question_random"))
        if self.method == "setpo":
            self.embedder = setpo.FullResponseEmbedder(kwargs["embedding_model"],
                device=kwargs["embedding_device"], batch_size=kwargs["embedding_batch_size"],
                max_length=kwargs["embedding_max_tokens"],
                cpu_dtype=kwargs.get("embedding_cpu_dtype", "float32"),
                cpu_threads=kwargs.get("embedding_cpu_threads", 2))

    def auxiliary_experience(self, row, step, index):
        if self.method in DPH_METHODS:
            tokens, length = row["tokens"], row["prompt_length"]
            logprobs = row["reference_logprobs"]
            if len(logprobs) != len(tokens) - length:
                raise ValueError("DPH frozen token/logprob alignment is invalid")
        else:
            tokens, length, _ = render_sft_with_terminal_eos(self.tokenizer, row["messages"])
            logprobs = [0.] * (len(tokens) - length)
        response_length = len(tokens) - length
        return Experience(tokens=tokens, prompt_length=length, logprobs=logprobs,
            reward=0., advantages=torch.zeros(response_length), returns=torch.zeros(response_length),
            action_mask=torch.ones(response_length, dtype=torch.bool),
            eid=EID(batch=step, task=f"replay-{index}"),
            info={"is_expert": True, "problem_id": row["metadata"]["problem_id"]})

    async def sample(self, step):
        args = self.args
        online = await self.reader.read()
        groups = fresh_groups(online, step, args["online_count"], args["repeat_times"])
        for group in groups:
            rewards = [e.reward for e in group]
            if (self.method in DPH_METHODS and args.get("dynamic_sampling", True)
                    and not 0 < sum(rewards) < len(rewards)):
                raise RuntimeError("DPH accepted a homogeneous group; dynamic sampling contract violated")
            if self.method == "maxrl":
                advantages = maxrl.advantages(rewards)
            elif self.method == "setpo":
                texts = [self.tokenizer.decode(e.tokens[e.prompt_length:].tolist(), skip_special_tokens=True) for e in group]
                advantages = setpo.advantages(rewards, self.embedder.encode(texts),
                                               args["setpo_weight"], args["setpo_tau"]).cpu()
            else:
                advantages = grpo.advantages(rewards)
            for exp, advantage in zip(group, advantages):
                exp.advantages = advantage * exp.action_mask.float()
                exp.returns = exp.advantages.clone()
                exp.info["is_expert"] = False
        auxiliary = [] if self.replay is None else [self.auxiliary_experience(row, step, i)
            for i, row in enumerate(self.replay.sample(args["replay_count"], step))]
        experiences = arrange(groups, auxiliary, prompts_per_update=args["prompts_per_update"],
                              dp_size=args["dp_size"], seed=args["seed"], iteration=step)
        compacted_tokens = 0
        if args.get("compact_zero_loss", False):
            compacted_tokens = sum(compact_zero_loss(exp) for exp in experiences)
        for exp in experiences:
            exp.custom_fields = [CustomField("is_expert", "expert_mask", torch.bool),
                CustomField("online_token_count", "online_token_count", torch.float32),
                CustomField("auxiliary_token_count", "auxiliary_token_count", torch.float32),
                CustomField("auxiliary_response_count", "auxiliary_response_count", torch.float32)]
        metrics = {"sample/compacted_zero_loss_tokens": compacted_tokens, "sample/online_count": len(online), "sample/replay_count": len(auxiliary),
            "sample/mixed_reward_group_fraction": sum(0 < sum(e.reward for e in g) < len(g) for g in groups) / len(groups),
            "sample/generated_responses": sum(e.info.get("generated_responses", 0) for e in online),
            "sample/generated_response_tokens": sum(e.info.get("generated_tokens", 0) for e in online),
            "sample/discarded_groups": sum(e.info.get("discarded_groups", 0) for e in online),
            "sample/discarded_all_wrong_groups": sum(e.info.get("discarded_all_wrong_groups", 0) for e in online),
            "sample/discarded_all_correct_groups": sum(e.info.get("discarded_all_correct_groups", 0) for e in online),
            "sample/replenishment_blocks": sum(e.info.get("replenishment_blocks", 0) for e in online),
            "sample/reward_mean": sum(e.reward for e in online) / len(online),
            "sample/truncation_fraction": sum(e.truncate_status == "response_truncated" for e in online) / len(online),
            "sample/optimizer_updates": len(groups) // args["prompts_per_update"]}
        self.set_model_version_metric(online, metrics)
        return experiences, metrics, representative_sample(experiences)

    @classmethod
    def default_args(cls):
        return {}

    def state_dict(self):
        return {"reader": self.reader.state_dict(), "replay_fingerprint": self.replay.fingerprint if self.replay else None}

    def load_state_dict(self, state):
        if state:
            if state["replay_fingerprint"] != (self.replay.fingerprint if self.replay else None):
                raise ValueError("Replay changed since checkpoint")
            self.reader.load_state_dict(state["reader"])


@POLICY_LOSS_FN.register_module("evorl_baselines")
class BaselineLoss(PolicyLossFn):
    def __init__(self, backend="verl", **kwargs):
        super().__init__(backend=backend)
        kwargs.pop("optimizer_batch_size", None)  # consumed by staged Trinity config
        self.options = kwargs

    def __call__(self, logprob, old_logprob, action_mask, advantages, expert_mask,
                 online_token_count, auxiliary_token_count, auxiliary_response_count=None):
        return objective.loss(logprob, old_logprob, action_mask, advantages, expert_mask,
                              online_token_count, auxiliary_token_count,
                              auxiliary_response_count=auxiliary_response_count, **self.options)

    @classmethod
    def default_args(cls):
        return {}
