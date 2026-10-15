"""Exact selected-token log-softmax with bounded temporary storage.

The fallback in pinned veRL keeps a vocabulary-sized softmax per token alive
for backward. Recompute those values in chunks instead. No tokens are removed,
and the input/output dtype and log-softmax expression remain unchanged.
"""
import torch
import torch.nn.functional as F


class _ChunkedLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, chunk_size):
        flat = logits.reshape(-1, logits.shape[-1])
        targets = labels.reshape(-1)
        result = torch.empty(targets.shape, device=logits.device, dtype=logits.dtype)
        for start in range(0, len(targets), chunk_size):
            end = start + chunk_size
            logp = F.log_softmax(flat[start:end], dim=-1)
            result[start:end] = logp.gather(-1, targets[start:end, None]).squeeze(-1)
        ctx.save_for_backward(logits, labels)
        ctx.chunk_size = chunk_size
        return result.reshape(labels.shape)

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels = ctx.saved_tensors
        flat = logits.reshape(-1, logits.shape[-1])
        targets, upstream = labels.reshape(-1), grad_output.reshape(-1)
        gradient = torch.empty_like(flat)
        for start in range(0, len(targets), ctx.chunk_size):
            end = start + ctx.chunk_size
            logp = F.log_softmax(flat[start:end], dim=-1)
            # The same log_softmax backward as PyTorch, without retaining its
            # full vocabulary output across the forward/backward boundary.
            selected_grad = torch.zeros_like(logp)
            selected_grad.scatter_(-1, targets[start:end, None], upstream[start:end, None])
            gradient[start:end] = torch.ops.aten._log_softmax_backward_data(
                selected_grad, logp, -1, logits.dtype)
        return gradient.reshape(logits.shape), None, None


def chunked_logprobs(logits, labels, chunk_size=128):
    if labels.shape != logits.shape[:-1] or chunk_size < 1:
        raise ValueError("Invalid selected-token log-probability shapes/chunk size")
    return _ChunkedLogProbs.apply(logits, labels, chunk_size)


def concatenate_logits(chunks):
    """Avoid copying a whole vocabulary tensor for a singleton microbatch.

    torch.cat([x]) allocates a copy even when x is already the exact unpadded
    sequence. Keeping its view preserves values and autograd while avoiding
    another sequence-length x vocabulary allocation during backward.
    """
    return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


def unpad_logits(logits, lengths):
    """Flatten right-padded logits without a vocabulary-sized jagged backward.

    A full singleton microbatch is just a reshape. The general path slices
    dense tensors; it retains exactly the same tokens and their gradients.
    """
    sizes = lengths.tolist()
    if len(sizes) != logits.shape[0] or any(n < 0 or n > logits.shape[1] for n in sizes):
        raise ValueError("Invalid padded logit lengths")
    if len(sizes) == 1 and sizes[0] == logits.shape[1]:
        return logits.reshape(-1, logits.shape[-1])
    return concatenate_logits([logits[i, :n] for i, n in enumerate(sizes)])
