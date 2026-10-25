"""Group count display and threshold evaluation (task 2).

Levels are computed from thresholds, with optional hysteresis so a count
hovering at a boundary does not flap between levels every frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

from .models import ThresholdLevel


@dataclass(frozen=True)
class Thresholds:
    """Warning/critical group-count thresholds for one region."""

    warning_at: int
    critical_at: int

    def __post_init__(self) -> None:
        if self.warning_at < 1:
            raise ValueError(f"warning_at must be >= 1, got {self.warning_at}")
        if self.critical_at < self.warning_at:
            raise ValueError(
                f"critical_at ({self.critical_at}) must be >= warning_at ({self.warning_at})"
            )

    def evaluate(self, count: int) -> ThresholdLevel:
        if count >= self.critical_at:
            return ThresholdLevel.CRITICAL
        if count >= self.warning_at:
            return ThresholdLevel.WARNING
        return ThresholdLevel.OK


@dataclass(frozen=True)
class RegionStatus:
    """One row of the group-count display."""

    region: str
    people_count: int
    group_count: int
    largest_group: int
    level: ThresholdLevel


_LEVEL_MARKS: Dict[ThresholdLevel, str] = {
    ThresholdLevel.OK: " ",
    ThresholdLevel.WARNING: "!",
    ThresholdLevel.CRITICAL: "!!",
}


def render_display(statuses: Sequence[RegionStatus], width: int = 68) -> str:
    """Render an ASCII table of per-region group counts and threshold levels."""
    lines = [
        "PEOPLE GROUP MONITOR".center(width, "="),
        f"  {'REGION':<18}{'PEOPLE':>7}{'GROUPS':>8}{'LARGEST':>9}   LEVEL",
        "-" * width,
    ]
    if not statuses:
        lines.append("(no monitored regions)")
    for status in statuses:
        mark = _LEVEL_MARKS[status.level]
        lines.append(
            f"{mark:<2}{status.region:<18}"
            f"{status.people_count:>7}{status.group_count:>8}{status.largest_group:>9}   "
            f"{status.level.value.upper()}"
        )
    lines.append("-" * width)

    total_people = sum(s.people_count for s in statuses)
    worst = max((s.level for s in statuses), key=lambda level: level.rank, default=ThresholdLevel.OK)
    lines.append(f"TOTAL people={total_people} regions={len(statuses)} status={worst.value.upper()}")
    lines.append("=" * width)
    return "\n".join(lines)


class ThresholdTracker:
    """Stateful level tracking with hysteresis, keyed by region.

    Escalation happens as soon as a threshold is crossed. De-escalation only
    happens once the count drops ``hysteresis`` below the threshold that was
    crossed, so jitter at a boundary does not spam events.
    """

    def __init__(self, hysteresis: int = 1) -> None:
        if hysteresis < 0:
            raise ValueError(f"hysteresis must be >= 0, got {hysteresis}")
        self._hysteresis = hysteresis
        self._levels: Dict[str, ThresholdLevel] = {}

    def level(self, key: str) -> ThresholdLevel:
        return self._levels.get(key, ThresholdLevel.OK)

    def update(self, key: str, count: int, thresholds: Thresholds) -> ThresholdLevel:
        current = self.level(key)
        if current == ThresholdLevel.OK:
            level = thresholds.evaluate(count)
        elif current == ThresholdLevel.WARNING:
            if count >= thresholds.critical_at:
                level = ThresholdLevel.CRITICAL
            elif count < thresholds.warning_at - self._hysteresis:
                level = ThresholdLevel.OK
            else:
                level = ThresholdLevel.WARNING
        else:  # CRITICAL
            if count >= thresholds.critical_at - self._hysteresis:
                level = ThresholdLevel.CRITICAL
            elif count >= thresholds.warning_at - self._hysteresis:
                level = ThresholdLevel.WARNING
            else:
                level = ThresholdLevel.OK
        self._levels[key] = level
        return level

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._levels.clear()
        else:
            self._levels.pop(key, None)
