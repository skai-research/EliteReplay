"""SetPO+GRPO: the authors' marginals, added after GRPO normalization.

The vendored reference and its license live beside this file. No centering,
correct-only filtering, reward replacement, or second standardization is used.
"""
from __future__ import annotations

import torch
from .grpo import advantages as grpo_advantages


def advantages(rewards, embeddings, weight=0.05, tau=1.0):
    from .setpo_reference import _facility_location_scores
    if len(rewards) < 3 or len(rewards) != embeddings.shape[0]:
        raise ValueError("SetPO requires at least three aligned responses")
    if weight < 0 or tau <= 0:
        raise ValueError("Invalid SetPO weight/tau")
    _, marginal = _facility_location_scores(embeddings, tau)
    if not torch.isfinite(marginal).all():
        raise FloatingPointError("Non-finite SetPO marginal")
    return grpo_advantages(rewards).to(marginal.device) + weight * marginal


class FullResponseEmbedder:
    """Use the reference encoder while rejecting silent embedding truncation."""

    def __init__(self, model_path, *, device="cpu", batch_size=16, max_length=32768,
                 cpu_dtype="float32", cpu_threads=2):
        from .setpo_reference import _TextEmbeddingModel
        self.encoder = _TextEmbeddingModel(model_path, device=device,
                                          batch_size=batch_size, max_length=max_length)
        if cpu_dtype not in ("float32", "bfloat16") or type(cpu_threads) is not int or cpu_threads < 1:
            raise ValueError("Invalid CPU embedding dtype or thread count")
        if self.encoder.device.type == "cpu":
            # BF16 CPU execution was much slower than FP32 in the runtime audit.
            # Preserve the encoder/pooling/marginal; only CPU arithmetic changes.
            self.encoder.compute_dtype = getattr(torch, cpu_dtype)
            torch.set_num_threads(cpu_threads)

    def encode(self, texts):
        self.encoder.warmup()
        if self.encoder.device.type == "cpu":
            self.encoder._model.to(dtype=self.encoder.compute_dtype)
        lengths = [len(ids) for ids in self.encoder._tokenizer(
            list(texts), truncation=False, padding=False)["input_ids"]]
        if lengths and max(lengths) > self.encoder.max_length:
            raise ValueError("SetPO embedding limit is shorter than a response; increase embedding_max_tokens")
        return self.encoder.encode(texts)
