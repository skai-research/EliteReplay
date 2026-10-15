"""SetPO reference components, vendored without algorithm changes.

Source: https://github.com/chenyili0818/SetPO
Commit: 9803ca9c2cb5a3a21e9af3a570044d4c90cf0be3
File: verl/trainer/main_ppo.py
License: Apache-2.0 (see SETPO_LICENSE). Only module imports/cache declaration
are adapted; the two definitions below are exact upstream source excerpts.
"""
from __future__ import annotations
import os
import json
import warnings
from contextlib import nullcontext
from typing import Sequence, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
_EMBEDDING_MODEL_CACHE = {}

class _TextEmbeddingModel:
    """Text embedding model used by the facility-location reward."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 1024,
        trust_remote_code: bool = False,
    ):
        resolved_id = os.path.expanduser(model_id)
        if os.path.isfile(resolved_id):
            resolved_id = os.path.dirname(resolved_id)

        self.model_id = resolved_id
        self.max_length = max_length
        self.batch_size = max(1, int(batch_size))
        self.trust_remote_code = trust_remote_code
        self.compute_dtype = torch.bfloat16

        self._local_files_only = os.path.isdir(resolved_id)
        self._pooling_mode = self._detect_pooling_mode()

        self._tokenizer: AutoTokenizer | None = None
        self._model: torch.nn.Module | None = None

        self.device = self._select_device(device)

    def __getstate__(self):
        """Drop heavyweight model state before pickling."""
        state = self.__dict__.copy()
        state["_tokenizer"] = None
        state["_model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _select_device(self, device_str: str) -> torch.device:
        dev = device_str.lower()

        if dev == "auto":
            if torch.cuda.is_available():
                print("TextEmbeddingModel: auto-selecting CUDA device.")
                return torch.device("cuda")
            warnings.warn(
                "TextEmbeddingModel: CUDA not available, falling back to CPU."
            )
            return torch.device("cpu")

        if dev.startswith("cuda"):
            if torch.cuda.is_available():
                print("TextEmbeddingModel: using CUDA device.")
                return torch.device(device_str)
            warnings.warn(
                f"TextEmbeddingModel: requested device '{device_str}' "
                "but no CUDA GPUs are available, falling back to CPU."
            )
            return torch.device("cpu")

        return torch.device(device_str)

    def _detect_pooling_mode(self) -> str:
        """Read the sentence-transformers pooling config when available."""
        pooling_config_path = os.path.join(self.model_id, "1_Pooling", "config.json")
        if not os.path.isfile(pooling_config_path):
            return "last_token"

        try:
            with open(pooling_config_path, "r", encoding="utf-8") as f:
                pooling_cfg = json.load(f)
        except Exception as exc:
            warnings.warn(
                f"TextEmbeddingModel: failed to read pooling config from "
                f"'{pooling_config_path}', falling back to last-token pooling: {exc}"
            )
            return "last_token"

        if pooling_cfg.get("pooling_mode_lasttoken", False):
            return "last_token"
        if pooling_cfg.get("pooling_mode_cls_token", False):
            return "cls"
        if pooling_cfg.get("pooling_mode_mean_tokens", False):
            return "mean"
        return "last_token"

    def _ensure_loaded(self) -> None:
        """Load the model for the current process on first use."""
        if self._tokenizer is not None and self._model is not None:
            return
        cache_key = (self.model_id, str(self.device))
        cached = _EMBEDDING_MODEL_CACHE.get(cache_key)

        if cached is None:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self._local_files_only,
            )
            tokenizer.padding_side = "left" if self._pooling_mode == "last_token" else "right"
            model = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self._local_files_only,
                torch_dtype=self.compute_dtype,
            )
            model.to(self.device)
            model.eval()

            _EMBEDDING_MODEL_CACHE[cache_key] = (tokenizer, model)
            cached = (tokenizer, model)

        self._tokenizer, self._model = cached

    def warmup(self) -> None:
        """Eagerly load the model."""
        self._ensure_loaded()

    @staticmethod
    def _last_token_pool(
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """last-token pooling。"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]

        seq_lens = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            seq_lens,
        ]

    @staticmethod
    def _mean_pool(
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """attention-mask mean pooling。"""
        mask = attention_mask.unsqueeze(-1).to(last_hidden_states.dtype)
        summed = (last_hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        return summed / denom

    def _pool(
        self,
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self._pooling_mode == "cls":
            return last_hidden_states[:, 0]
        if self._pooling_mode == "mean":
            return self._mean_pool(last_hidden_states, attention_mask)
        return self._last_token_pool(last_hidden_states, attention_mask)

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode a batch of strings into normalized embeddings."""
        if len(texts) == 0:
            return torch.empty((0, 1), device=self.device)

        self._ensure_loaded()

        all_embeddings: list[torch.Tensor] = []
        bs = self.batch_size

        for start in range(0, len(texts), bs):
            chunk = list(texts[start : start + bs])

            inputs = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=self.compute_dtype)
                if self.device.type == "cuda" and self.compute_dtype in (torch.float16, torch.bfloat16)
                else nullcontext()
            )

            with torch.no_grad(), amp_ctx:
                outputs = self._model(**inputs)

            pooled = self._pool(
                outputs.last_hidden_state,
                inputs["attention_mask"],
            )
            embeddings = F.normalize(pooled, p=2, dim=1)   # [b, D]
            all_embeddings.append(embeddings.detach())

        return torch.cat(all_embeddings, dim=0)

def _facility_location_scores(
    embeddings: torch.Tensor,
    tau: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute facility-location marginals for a batch of embeddings.

        D_1(S) = \sum_{u\in A} g\Bigl(\sum_{j\in S,\,j\neq u} \text{sim}(e_j,e_u)\Bigr),
        g(x) = log(1 + x / tau),

    When S = A, the per-item marginal is:
        s_k = D_1(A) - D_1(A \setminus \{k\})
            = \sum_{u} g(z_u) - g(z_u - sim(e_k, e_u)).

    Returns:
        selected: [N] bool
        marginals: [N] float
    """
    if embeddings.numel() == 0:
        device = embeddings.device
        empty_bool = torch.empty(0, dtype=torch.bool, device=device)
        empty_float = torch.empty(0, device=device, dtype=embeddings.dtype)
        zero = torch.tensor(0.0, device=device, dtype=embeddings.dtype)
        return empty_bool, empty_float

    device = embeddings.device
    N = embeddings.size(0)
    b = embeddings.size(1)
    print("computing SetPO rewards for batch size =", N)

    emb = embeddings.to(dtype=torch.float32)
    emb = F.normalize(emb, p=2, dim=-1)

    sim = emb @ emb.T
    # sim.exp_()
    sim.clamp_min_(0.0)
    sim.fill_diagonal_(0.0)

    # z_u = sum_j sim[j, u]
    z = sim.sum(dim=0)  # [N]
    z_mean = z / (N - 1)

    # g(x) = log(1 + x / tau)
    def g(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=0.0)
        return - torch.log1p(x / tau)

    # D_1(A) = sum_u g(z_u)
    g_z = g(z_mean)                 # [N]
    total_g = g_z.sum()

    # z_minus[k, u] = z_u - sim[k, u]
    z_minus = z.unsqueeze(0) - sim   # [N, N]
    # z_minus.clamp_min_(0.0)
    z_minus_mean = z_minus / (N - 2)
    g_z_minus = g(z_minus_mean)           # [N, N]

    # s_k = total_g - sum_u g(z_u - sim[k, u])
    # marginals = total_g - g_z_minus.sum(dim=1)   # [N]
    marginals = total_g / N - g_z_minus.sum(dim=1) / (N - 1)
    marginals = marginals.to(dtype=torch.float32)

    if marginals.numel() > 0:
        scores_mean = marginals.mean()
        scores_var = marginals.var(unbiased=False)
    else:
        scores_mean = torch.tensor(0.0, device=device, dtype=torch.float32)
        scores_var = torch.tensor(0.0, device=device, dtype=torch.float32)
        
    print("current batch SetPO reward stats: mean =", scores_mean.item())
    print("current batch SetPO reward stats: var =", scores_var.item())

    selected = torch.ones(N, dtype=torch.bool, device=device)

    return selected, marginals
