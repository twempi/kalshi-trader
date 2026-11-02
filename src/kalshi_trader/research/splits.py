"""Train / validation / test splits that never leak an event across groups.

Many observations come from the same match (event). If observations were split at random,
or by timestamp alone, one match could contribute to both training and test data and make
results look better than they are. Here whole *events* are assigned to splits by the time
they were first observed, in chronological order, with an embargo: events first observed
within ``embargo_s`` after a boundary are dropped (including the one that defines the
boundary), so a split never contains matches that overlapped in time with the end of the
previous split (same day, same tournament conditions, same market regime).

Rules for use (enforced by the replay CLI, documented in the README):

* choose parameters on ``train`` (and ``validation`` for model selection);
* evaluate ``test`` once per configuration; every evaluation is logged with its config hash
  so repeated peeking is visible in the report.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EventSplit:
    train: frozenset[str]
    validation: frozenset[str]
    test: frozenset[str]
    boundaries: tuple[float, float]
    dropped_by_embargo: frozenset[str] = field(default_factory=frozenset)

    def group_of(self, event: str) -> str | None:
        for name in ("train", "validation", "test"):
            if event in getattr(self, name):
                return name
        return None

    def summary(self) -> dict[str, object]:
        return {
            "train_events": len(self.train),
            "validation_events": len(self.validation),
            "test_events": len(self.test),
            "embargo_dropped": len(self.dropped_by_embargo),
            "boundaries": list(self.boundaries),
        }


def split_events(
    first_seen: Mapping[str, float],
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.3,
    embargo_s: float = 3600.0,
) -> EventSplit:
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("fractions must be in [0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation + test must leave room for training data")
    ordered = sorted(first_seen.items(), key=lambda kv: (kv[1], kv[0]))
    n = len(ordered)
    if n == 0:
        return EventSplit(frozenset(), frozenset(), frozenset(), (0.0, 0.0))
    i_val = round(n * (1 - validation_fraction - test_fraction))
    i_test = round(n * (1 - test_fraction))
    ts_val = ordered[i_val][1] if i_val < n else float("inf")
    ts_test = ordered[i_test][1] if i_test < n else float("inf")
    train, val, test, dropped = set(), set(), set(), set()
    for index, (event, ts) in enumerate(ordered):
        if index < i_val:
            train.add(event)
        elif index < i_test:
            (dropped if ts < ts_val + embargo_s else val).add(event)
        else:
            (dropped if ts < ts_test + embargo_s else test).add(event)
    return EventSplit(
        frozenset(train),
        frozenset(val),
        frozenset(test),
        (ts_val, ts_test),
        frozenset(dropped),
    )
