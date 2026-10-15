"""Project-local Trinity-RFT integration for EliteReplay.

Load this module with ``trinity run --plugin-dir plugins``.  It intentionally
lives outside the copied Trinity-RFT checkout so that upstream code remains a
pinned, inspectable dependency.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import math
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import torch

from elitereplay.archive.trinity_support import (
    FROZEN_SFT_SOURCE,
    FROZEN_SFT_SOURCE_ID,
    ON_POLICY_SOURCE,
    ON_POLICY_SOURCE_ID,
    UniformProblemNicheArchive,
    render_sft_with_terminal_eos,
    repeated_special_tag,
)
from trinity.algorithm.policy_loss_fn import POLICY_LOSS_FN
from trinity.algorithm.policy_loss_fn.chord_policy_loss import MIXCHORDPolicyLossFn
from trinity.algorithm.sample_strategy import SAMPLE_STRATEGY, SampleStrategy
from trinity.algorithm.sample_strategy.utils import representative_sample
from trinity.buffer import get_buffer_reader
from trinity.buffer.reader import READER
from trinity.buffer.reader.file_reader import TaskFileReader, _HFBatchReader
from trinity.buffer.schema import FORMATTER
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.experience import CustomField, EID, Experience
from trinity.common.workflows import WORKFLOWS, Task, Workflow
from trinity.utils.timer import Timer


def _load_dataset(path: str, *, split: str = "train", subset_name: str | None = None):
    """Load local JSONL/JSON/Parquet directly, with HF datasets as fallback."""

    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

    source = Path(path)
    if source.is_file():
        suffix = source.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            return Dataset.from_json(str(source))
        if suffix == ".parquet":
            return Dataset.from_parquet(str(source))
        if suffix == ".csv":
            return Dataset.from_csv(str(source))
        raise ValueError(f"unsupported local EliteReplay dataset type: {source}")
    if source.is_dir() and (
        (source / "state.json").exists() or (source / "dataset_dict.json").exists()
    ):
        dataset = load_from_disk(str(source))
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                raise ValueError(f"dataset {source} has no split {split!r}")
            return dataset[split]
        return dataset
    return load_dataset(path, name=subset_name, split=split)


@READER.register_module("EliteReplay_file")
class EliteReplayTaskFileReader(TaskFileReader):
    """Trinity task reader that supports the pipeline's direct local files."""

    def _init_formatter_and_dataset(self, config: StorageConfig):
        formatter = FORMATTER.get("task")(config)
        raw_dataset = _load_dataset(
            str(config.path), split=config.split, subset_name=config.subset_name
        )
        dataset = _HFBatchReader(
            raw_dataset,
            name=config.name,
            default_batch_size=self.read_batch_size,
            total_epochs=config.total_epochs if not config.is_eval else 1,
            offset=config.index,
            drop_last=not config.is_eval,
            total_steps=config.total_steps if not config.is_eval else None,
            enable_progress_bar=config.enable_progress_bar,
        )
        return formatter, dataset


@FORMATTER.register_module("EliteReplay_sft")
class EliteReplaySFTFormatter:
    """Render a canonical executable response with one supervised real EOS."""

    def __init__(self, tokenizer_path: str, format_config: Any):
        from trinity.common.models.mm_utils import processor_or_tokenizer_cls

        processor_cls = processor_or_tokenizer_cls(tokenizer_path)
        self.processor = processor_cls.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.messages_key = format_config.messages_key
        self.chat_template = format_config.chat_template
        self.enable_thinking = format_config.enable_thinking

    def format(self, sample: Mapping[str, Any]) -> Experience:
        metadata_value = sample.get("metadata", {})
        if hasattr(metadata_value, "as_py"):
            metadata_value = metadata_value.as_py()
        if not isinstance(metadata_value, Mapping):
            raise ValueError("SFT metadata must be a mapping")
        metadata = dict(metadata_value)
        existing_source = metadata.get("sample_source", sample.get("sample_source"))
        if existing_source not in (None, FROZEN_SFT_SOURCE):
            raise ValueError(f"SFT row has conflicting sample_source={existing_source!r}")

        tokens, prompt_length, messages = render_sft_with_terminal_eos(
            self.tokenizer,
            sample[self.messages_key],
            chat_template=self.chat_template,
            enable_thinking=self.enable_thinking,
        )
        problem_id = str(metadata.get("problem_id", sample.get("problem_id", "")))
        candidate_id = str(metadata.get("candidate_id", sample.get("candidate_id", "")))
        primary = str(
            metadata.get("primary_strategy", sample.get("primary_strategy", ""))
        )
        secondary_value = metadata.get(
            "secondary_strategy", sample.get("secondary_strategy")
        )
        secondary = "" if secondary_value in (None, "") else str(secondary_value)
        if not problem_id or not candidate_id or not primary:
            raise ValueError("SFT metadata requires problem_id, candidate_id, and primary_strategy")

        response_length = len(tokens) - prompt_length
        info = {
            "sample_source": FROZEN_SFT_SOURCE,
            "sample_source_id": FROZEN_SFT_SOURCE_ID,
            "is_expert": True,
            "problem_id": problem_id,
            "candidate_id": candidate_id,
            "primary_strategy": primary,
            "secondary_strategy": secondary,
            "archive_method": str(metadata.get("archive_method", "")),
        }
        return Experience(
            eid=EID(
                batch=FROZEN_SFT_SOURCE,
                task=problem_id,
                run=0,
                step=0,
                suffix=candidate_id[:12],
            ),
            tokens=tokens,
            prompt_length=prompt_length,
            response_text=messages[-1]["content"],
            messages=messages,
            info=info,
            metrics={
                "sft/response_tokens": float(response_length),
                "sft/terminal_eos": 1.0,
            },
        )


def _domain_adapter(domain: str, adapter_args: Mapping[str, Any]):
    """Import domain code only when a workflow is instantiated by a runner."""

    domain = domain.lower().strip()
    if domain not in {"bird", "taco", "optmath"}:
        raise ValueError(f"unsupported EliteReplay domain: {domain!r}")
    module = importlib.import_module(f"elitereplay.archive.{domain}")
    class_name = {"bird": "BirdAdapter", "taco": "TacoAdapter", "optmath": "OptmathAdapter"}[domain]
    adapter_cls = getattr(module, class_name)
    return adapter_cls(**dict(adapter_args))


@WORKFLOWS.register_module("EliteReplay_workflow")
class EliteReplayWorkflow(Workflow):
    """Single-turn, strict-verifier workflow used only for fresh GRPO rollouts."""

    can_reset = True
    can_repeat = True

    def __init__(self, *, task: Task, model: Any, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.reset(task)

    def reset(self, task: Task):
        self.task = task
        self.raw_task = task.raw_task
        self.workflow_args = dict(task.workflow_args or {})
        self.adapter = _domain_adapter(
            str(self.workflow_args.get("domain", "")),
            self.workflow_args.get("adapter_args", {}),
        )
        self.repeat_times = int(task.repeat_times or 1)

    def set_repeat_times(self, repeat_times: int, run_id_base: int) -> None:
        self.repeat_times = int(repeat_times)
        self.task.rollout_args.n = self.repeat_times
        self.run_id_base = int(run_id_base)

    @property
    def rollout_args(self) -> dict[str, Any]:
        return asdict(self.task.rollout_args)

    def run(self) -> list[Experience]:
        messages = self.adapter.format_prompt(self.raw_task)
        if not isinstance(messages, list):
            raise TypeError("DomainAdapter.format_prompt must return a message list")
        responses = self.model.chat(messages, **self.rollout_args)
        if len(responses) != self.repeat_times:
            raise RuntimeError(
                f"rollout engine returned {len(responses)} responses; "
                f"expected {self.repeat_times}"
            )
        for index, response in enumerate(responses):
            self._score_response(response, index)
        return responses

    def _score_response(self, response: Experience, index: int) -> None:
        if response.info is None:
            response.info = {}
        old_source = response.info.get("sample_source")
        if old_source not in (None, ON_POLICY_SOURCE):
            raise RuntimeError(f"rollout arrived with conflicting source {old_source!r}")
        if response.info.get("is_expert") is True:
            raise RuntimeError("fresh rollout was marked as expert data")

        raw_response = response.response_text or ""
        truncated = response.truncate_status == "response_truncated"
        parse_result = self.adapter.parse_strict(raw_response)
        verification = None
        if parse_result.accepted and not truncated:
            verification = self.adapter.verify(
                self.raw_task, parse_result.canonical_response
            )
        reward = float(verification.score) if verification is not None else 0.0
        if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise RuntimeError(f"domain verifier returned invalid reward {reward!r}")

        canonical = parse_result.canonical_response or ""
        response.reward = reward
        response.eid.run = index + self.run_id_base
        response.info.update(
            {
                "sample_source": ON_POLICY_SOURCE,
                "sample_source_id": ON_POLICY_SOURCE_ID,
                "is_expert": False,
                "strict_parse": bool(parse_result.accepted and not truncated),
                "executable": bool(verification and verification.executable),
                "fully_correct": bool(verification and verification.fully_correct),
                "canonical_response_sha256": hashlib.sha256(
                    canonical.encode("utf-8")
                ).hexdigest()
                if canonical
                else "",
                "parse_error": parse_result.error or "",
                "verification_error": verification.error
                if verification and verification.error
                else "",
            }
        )
        if response.metrics is None:
            response.metrics = {}
        response.metrics.update(
            {
                "reward/verification": reward,
                "safety/strict_parse": float(parse_result.accepted and not truncated),
                "safety/executable": float(bool(verification and verification.executable)),
                "safety/fully_correct": float(
                    bool(verification and verification.fully_correct)
                ),
                "safety/response_truncated": float(truncated),
                "safety/forbidden_special_tag": float(
                    repeated_special_tag(raw_response)
                ),
            }
        )
        if truncated and response.action_mask is not None:
            response.action_mask = torch.zeros_like(response.action_mask, dtype=torch.bool)


def _percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[max(index, 0)])


def _response_lengths(experiences: Sequence[Experience]) -> list[int]:
    return [int(len(exp.tokens) - exp.prompt_length) for exp in experiences]


@SAMPLE_STRATEGY.register_module("EliteReplay_mix")
class EliteReplayMixSampleStrategy(SampleStrategy):
    """Mix fresh GRPO and frozen SFT with true problem-then-niche replay."""

    def __init__(self, buffer_config: BufferConfig, **kwargs):
        super().__init__(buffer_config)
        total_batch_size = int(buffer_config.train_batch_size)
        self.expert_batch_size = int(kwargs.get("expert_batch_size", 0))
        if self.expert_batch_size <= 0:
            ratio = float(kwargs.get("expert_data_ratio", 0.5))
            self.expert_batch_size = int(round(ratio * total_batch_size))
        self.usual_batch_size = total_batch_size - self.expert_batch_size
        if self.expert_batch_size <= 0 or self.usual_batch_size <= 0:
            raise ValueError("EliteReplay requires non-empty GRPO and SFT sub-batches")
        expected_usual = kwargs.get("expected_usual_batch_size")
        if expected_usual is not None and int(expected_usual) != self.usual_batch_size:
            raise ValueError(
                f"usual batch is {self.usual_batch_size}, expected {expected_usual}"
            )

        usual_config = copy.deepcopy(buffer_config.trainer_input.experience_buffer)
        usual_config.batch_size = self.usual_batch_size
        self.usual_exp_buffer = get_buffer_reader(usual_config)

        dataset_name = str(kwargs.get("sft_dataset_name", "sft_dataset"))
        auxiliary = buffer_config.trainer_input.auxiliary_buffers or {}
        if dataset_name not in auxiliary:
            raise ValueError(f"auxiliary buffer {dataset_name!r} is not configured")
        expert_config = copy.deepcopy(auxiliary[dataset_name])
        if expert_config.schema_type != "EliteReplay_sft":
            raise ValueError(
                f"{dataset_name} schema_type must be 'EliteReplay_sft', "
                f"got {expert_config.schema_type!r}"
            )
        raw_dataset = _load_dataset(
            str(expert_config.path),
            split=expert_config.split,
            subset_name=expert_config.subset_name,
        )
        rows = [raw_dataset[index] for index in range(len(raw_dataset))]
        self.archive = UniformProblemNicheArchive(
            rows,
            seed=int(kwargs.get("seed", 42)),
            expected_archive_method=kwargs.get("expected_archive_method", "qd"),
            require_fully_correct=bool(kwargs.get("require_fully_correct", True)),
        )
        self.expert_formatter = FORMATTER.get("EliteReplay_sft")(
            tokenizer_path=expert_config.tokenizer_path,
            format_config=expert_config.format,
        )
        self.require_model_version = bool(kwargs.get("require_model_version", True))
        self.max_model_version_lag = int(kwargs.get("max_model_version_lag", 0))
        if self.max_model_version_lag < 0:
            raise ValueError("max_model_version_lag cannot be negative")

    async def sample(self, step: int):
        metrics: dict[str, float] = {}
        with Timer(metrics, "time/read_experience"):
            usual = await self.usual_exp_buffer.read()
            if len(usual) != self.usual_batch_size:
                raise RuntimeError(
                    f"GRPO buffer returned {len(usual)} samples; "
                    f"expected {self.usual_batch_size}"
                )
            self._prepare_usual(usual, step)
            records = self.archive.sample(self.expert_batch_size)
            expert = [self.expert_formatter.format(record.row) for record in records]
            self._prepare_expert(expert, records, step)
            experiences = usual + expert
            representative = representative_sample(experiences)

        self.set_model_version_metric(usual, metrics)
        usual_lengths = _response_lengths(usual)
        expert_lengths = _response_lengths(expert)
        problem_counts, niche_counts = self.archive.batch_counts(records)
        metrics.update(
            {
                "sample/on_policy_count": float(len(usual)),
                "sample/frozen_sft_count": float(len(expert)),
                "sample/frozen_sft_fraction": float(len(expert) / len(experiences)),
                "sample/frozen_sft_unique_problems": float(len(problem_counts)),
                "sample/frozen_sft_unique_problem_niches": float(len(niche_counts)),
                "sample/frozen_sft_archive_problems": float(self.archive.problem_count),
                "sample/frozen_sft_archive_niches": float(self.archive.niche_count),
                "sample/frozen_sft_cursor": float(self.archive.cursor),
                "sample/on_policy_response_tokens": float(sum(usual_lengths)),
                "sample/frozen_sft_response_tokens": float(sum(expert_lengths)),
                "sample/on_policy_response_length_mean": float(mean(usual_lengths)),
                "sample/on_policy_response_length_median": float(median(usual_lengths)),
                "sample/on_policy_response_length_p95": _percentile(usual_lengths, 0.95),
                "sample/on_policy_response_length_max": float(max(usual_lengths)),
                "sample/frozen_sft_response_length_mean": float(mean(expert_lengths)),
                "sample/frozen_sft_response_length_median": float(median(expert_lengths)),
                "sample/frozen_sft_response_length_p95": _percentile(expert_lengths, 0.95),
                "sample/frozen_sft_response_length_max": float(max(expert_lengths)),
                "sample/frozen_sft_eos_rate": 1.0,
                "sample/on_policy_truncated_fraction": float(
                    sum(exp.truncate_status == "response_truncated" for exp in usual)
                    / len(usual)
                ),
            }
        )
        for metric_name in (
            "safety/strict_parse",
            "safety/executable",
            "safety/forbidden_special_tag",
        ):
            values = [exp.metrics.get(metric_name) for exp in usual if metric_name in exp.metrics]
            if values:
                metrics[f"sample/on_policy_{metric_name.split('/')[-1]}_rate"] = float(
                    mean(values)
                )
        return experiences, metrics, representative

    def _prepare_usual(self, experiences: Sequence[Experience], step: int) -> None:
        expected_version = step - 1
        custom_fields = self._custom_fields()
        for exp in experiences:
            info = exp.info or {}
            if info.get("sample_source") != ON_POLICY_SOURCE:
                raise RuntimeError(
                    "GRPO buffer contained a sample that was not labeled "
                    f"{ON_POLICY_SOURCE!r}: {info.get('sample_source')!r}"
                )
            if info.get("is_expert") is True:
                raise RuntimeError("expert example reached the GRPO source buffer")
            if self.require_model_version:
                if "model_version" not in info:
                    raise RuntimeError("GRPO rollout is missing model_version")
                version = int(info["model_version"])
                drift = int(info.get("model_version_drift", 0))
                if drift != 0:
                    raise RuntimeError(
                        f"rollout policy changed during generation (drift={drift})"
                    )
                if not expected_version - self.max_model_version_lag <= version <= expected_version:
                    raise RuntimeError(
                        f"off-policy GRPO rollout at train step {step}: "
                        f"model_version={version}, expected {expected_version} "
                        f"with max lag {self.max_model_version_lag}"
                    )
            info["sample_source_id"] = ON_POLICY_SOURCE_ID
            info["is_expert"] = False
            info["step"] = step
            exp.info = info
            exp.custom_fields = custom_fields

    def _prepare_expert(self, experiences, records, step: int) -> None:
        custom_fields = self._custom_fields()
        for exp, record in zip(experiences, records):
            info = exp.info or {}
            if info.get("sample_source") != FROZEN_SFT_SOURCE:
                raise RuntimeError("SFT formatter violated the frozen source contract")
            if "model_version" in info:
                raise RuntimeError("frozen SFT example must not carry a rollout model version")
            response_tokens = exp.tokens[exp.prompt_length :]
            exp.reward = 0.0
            exp.logprobs = torch.zeros_like(response_tokens, dtype=torch.float32)
            exp.advantages = torch.zeros_like(response_tokens, dtype=torch.float32)
            info.update(
                {
                    "sample_source_id": FROZEN_SFT_SOURCE_ID,
                    "is_expert": True,
                    "step": step,
                    "archive_fingerprint": self.archive.fingerprint,
                    "problem_id": record.problem_id,
                    "candidate_id": record.candidate_id,
                }
            )
            exp.info = info
            exp.custom_fields = custom_fields

    @staticmethod
    def _custom_fields() -> list[CustomField]:
        return [
            CustomField("is_expert", "expert_mask", torch.bool),
            CustomField("sample_source_id", "sample_source_id", torch.int8),
            CustomField("step", "step", torch.int32),
        ]

    @classmethod
    def default_args(cls) -> dict[str, Any]:
        return {
            "expert_batch_size": 0,
            "expert_data_ratio": 0.5,
            "expected_usual_batch_size": None,
            "sft_dataset_name": "sft_dataset",
            "expected_archive_method": "qd",
            "require_fully_correct": True,
            "seed": 42,
            "require_model_version": True,
            "max_model_version_lag": 0,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "usual_buffer": self.usual_exp_buffer.state_dict(),
            "expert_sampler": self.archive.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not state_dict:
            return
        if int(state_dict.get("version", 1)) != 1:
            raise ValueError("unsupported EliteReplay sample strategy state")
        usual_state = state_dict.get("usual_buffer")
        if usual_state:
            self.usual_exp_buffer.load_state_dict(usual_state)
        expert_state = state_dict.get("expert_sampler")
        if expert_state:
            self.archive.load_state_dict(expert_state)


@POLICY_LOSS_FN.register_module("EliteReplay_mix")
class EliteReplayPolicyLoss(MIXCHORDPolicyLossFn):
    """CHORD loss with a tensor-level assertion of the source/mask boundary."""

    def __call__(
        self,
        *,
        logprob: torch.Tensor,
        old_logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        expert_mask: torch.Tensor,
        step: torch.Tensor,
        sample_source_id: torch.Tensor,
        **kwargs,
    ):
        source_is_expert = sample_source_id.to(torch.int64) == FROZEN_SFT_SOURCE_ID
        if not torch.equal(expert_mask.bool(), source_is_expert):
            raise RuntimeError("sample_source_id and expert_mask disagree")
        if torch.any(advantages[expert_mask.bool()] != 0):
            raise RuntimeError("frozen SFT examples entered the GRPO advantage path")
        loss, metrics = super().__call__(
            logprob=logprob,
            old_logprob=old_logprob,
            action_mask=action_mask,
            expert_mask=expert_mask,
            advantages=advantages,
            step=step,
            **kwargs,
        )
        metrics.update(
            {
                "source/on_policy_count": int((~expert_mask.bool()).sum().item()),
                "source/frozen_sft_count": int(expert_mask.bool().sum().item()),
            }
        )
        return loss, metrics
