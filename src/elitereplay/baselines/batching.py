"""DP-major batches with exact per-minibatch, per-source token normalization.

The pinned engine dispatches consecutive chunks to ranks, then splits each
rank's chunk into optimizer minibatches without shuffling. This layout keeps
every prompt group intact and gives each optimizer update its configured
online/replay counts. Dynamic microbatching can subsequently reorder within a
minibatch; the normalizers travel with each experience.
"""
from __future__ import annotations

import random


def arrange(groups, auxiliary, *, prompts_per_update, dp_size, seed, iteration):
    if len(groups) % prompts_per_update or prompts_per_update % dp_size:
        raise ValueError("Prompt counts must divide optimizer batches and trainer ranks")
    updates = len(groups) // prompts_per_update
    if len(auxiliary) % (updates * dp_size):
        raise ValueError("Replay count must divide updates and trainer ranks")
    groups = list(groups)
    random.Random(f"{seed}:batch:{iteration}").shuffle(groups)
    group_count = prompts_per_update // dp_size
    replay_count = len(auxiliary) // (updates * dp_size)
    partitions = [[None for _ in range(updates)] for _ in range(dp_size)]
    gi = ai = 0
    for rank in range(dp_size):
        for update in range(updates):
            fresh = [e for g in groups[gi:gi + group_count] for e in g]
            partitions[rank][update] = fresh + auxiliary[ai:ai + replay_count]
            gi += group_count
            ai += replay_count
    for update in range(updates):
        rows = [e for rank in partitions for e in rank[update]]
        counts = [sum(int(e.action_mask.sum()) for e in rows if bool(e.info["is_expert"]) == source)
                  for source in (False, True)]
        if counts[0] == 0 or (auxiliary and counts[1] == 0):
            raise ValueError("Optimizer batch has no active tokens for a required source")
        auxiliary_responses = max(1, sum(bool(e.info["is_expert"]) for e in rows))
        for exp in rows:
            exp.info.update(online_token_count=counts[0], auxiliary_token_count=max(1, counts[1]),
                            auxiliary_response_count=auxiliary_responses,
                            optimizer_batch=update)
    return [e for rank in partitions for batch in rank for e in batch]


def compact_zero_loss(experience):
    """Keep a two-token placeholder for a fully masked text-only online row.

    Call only after reward advantages and global denominators are computed.
    Preserve row/group positions and reward metadata; never compact replay.
    """
    if experience.info.get('is_expert') or bool(experience.action_mask.any()):
        return 0
    if getattr(experience, 'multi_modal_inputs', None):
        raise ValueError('Zero-loss compaction supports text-only experiences')
    import torch
    before = len(experience.tokens)
    start = experience.prompt_length
    if not 0 < start < before:
        raise ValueError('Expected a nonempty prompt and response')
    experience.info['zero_loss_original_tokens'] = before
    experience.tokens = experience.tokens[start - 1:start + 1].clone()
    experience.prompt_length = 1
    for name in ('logprobs', 'advantages', 'returns', 'action_mask'):
        value = getattr(experience, name)
        if value is not None:
            setattr(experience, name, value[:1].clone())
    # The old-probability value has no effect under a zero mask. A finite zero
    # avoids unnecessary probability ratios against the removed context.
    experience.logprobs.zero_()
    experience.advantages.zero_()
    experience.returns.zero_()
    experience.action_mask.zero_()
    return before - 2
