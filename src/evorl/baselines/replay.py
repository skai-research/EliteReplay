"""Identical question-balanced replay for QD and independent-sampling archives."""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict


class Replay:
    def __init__(self, rows, *, method, seed, sampling="question_random"):
        self.seed = seed
        if sampling not in ("question_random", "mode_cycles"):
            raise ValueError("Unknown replay sampling policy")
        self.sampling = sampling
        self.groups = defaultdict(list)
        self.modes = defaultdict(lambda: defaultdict(list))
        seen = set()
        material = []
        for row in rows:
            meta = row["metadata"]
            pid = str(meta["problem_id"])
            cid = str(meta["candidate_id"])
            if cid in seen:
                raise ValueError("Duplicate replay candidate ID")
            if meta.get("archive_method") != method or meta.get("fully_correct") is not True:
                raise ValueError("Replay provenance/correctness does not match this method")
            seen.add(cid)
            self.groups[pid].append(row)
            mode = meta.get("mode_id")
            if sampling == "mode_cycles" and not isinstance(mode, str):
                raise ValueError("Mode-balanced replay requires exported mode_id metadata")
            self.modes[pid][mode].append(row)
            material.append(row)
        if not self.groups:
            raise ValueError("Replay dataset is empty")
        self.fingerprint = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
        self.ids = sorted(self.groups)

    def _permutation(self, values, key):
        values = list(values)
        random.Random(f"{self.seed}:replay_cycles:{key}").shuffle(values)
        return values

    def sample(self, count, iteration):
        if count < 0 or iteration < 1:
            raise ValueError("Replay requires a nonnegative count and a one-based iteration")
        if self.sampling == "mode_cycles":
            # Every question occurs once in a shuffled cycle. Its successive
            # visits cycle through modes, and through representatives within
            # each mode. Absolute positions make resumed draws reproducible.
            chosen = []
            for position in range((iteration - 1) * count, iteration * count):
                question_cycle, offset = divmod(position, len(self.ids))
                pid = self._permutation(self.ids, f"questions:{question_cycle}")[offset]
                modes = sorted(self.modes[pid])
                mode_cycle, mode_offset = divmod(question_cycle, len(modes))
                mode = self._permutation(modes, f"modes:{pid}:{mode_cycle}")[mode_offset]
                teachers = sorted(self.modes[pid][mode], key=lambda r: r["metadata"]["candidate_id"])
                teacher_cycle, teacher_offset = divmod(mode_cycle, len(teachers))
                chosen.append(self._permutation(teachers, f"teachers:{pid}:{mode}:{teacher_cycle}")[teacher_offset])
            return chosen
        # Stateless per-iteration draw makes exact checkpoint replay independent
        # of how far an interrupted worker advanced after its last checkpoint.
        rng = random.Random(f"{self.seed}:replay:{iteration}")
        return [rng.choice(self.groups[rng.choice(self.ids)]) for _ in range(count)]
