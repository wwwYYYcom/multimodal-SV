from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler

from .session_trials import fisher_session_id


class SessionBatchSampler(Sampler[list[int]]):
    """Yield K utterance indices from one speaker-session per training item."""

    def __init__(
        self,
        rows: Sequence[dict[str, str]],
        utterances_per_session: int = 4,
        seed: int = 2027,
        shuffle: bool = True,
    ) -> None:
        if utterances_per_session <= 0:
            raise ValueError("utterances_per_session must be positive")
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            session_id = row.get("session_id") or fisher_session_id(row)
            groups[(row["speaker_id"], session_id)].append(index)
        self.groups = [
            indices
            for _, indices in sorted(groups.items())
            if len(indices) >= utterances_per_session
        ]
        if not self.groups:
            raise ValueError("no session has enough utterances")
        self.utterances_per_session = int(utterances_per_session)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        group_order = list(range(len(self.groups)))
        if self.shuffle:
            rng.shuffle(group_order)
        for group_index in group_order:
            indices = self.groups[group_index]
            if self.shuffle:
                yield rng.sample(indices, self.utterances_per_session)
            else:
                yield indices[:self.utterances_per_session]
