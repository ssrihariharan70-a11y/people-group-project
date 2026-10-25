"""Tests for group count display and thresholds (task 2)."""

import pytest

from people_group.models import ThresholdLevel
from people_group.thresholds import RegionStatus, Thresholds, ThresholdTracker, render_display


class TestThresholdEvaluation:
    def test_boundaries(self):
        thresholds = Thresholds(warning_at=3, critical_at=5)
        assert thresholds.evaluate(0) == ThresholdLevel.OK
        assert thresholds.evaluate(2) == ThresholdLevel.OK
        assert thresholds.evaluate(3) == ThresholdLevel.WARNING
        assert thresholds.evaluate(4) == ThresholdLevel.WARNING
        assert thresholds.evaluate(5) == ThresholdLevel.CRITICAL
        assert thresholds.evaluate(50) == ThresholdLevel.CRITICAL

    def test_warning_equals_critical_allowed(self):
        thresholds = Thresholds(warning_at=4, critical_at=4)
        assert thresholds.evaluate(4) == ThresholdLevel.CRITICAL

    def test_invalid_thresholds(self):
        with pytest.raises(ValueError, match="warning_at"):
            Thresholds(warning_at=0, critical_at=5)
        with pytest.raises(ValueError, match="critical_at"):
            Thresholds(warning_at=5, critical_at=3)


class TestThresholdTracker:
    def test_escalates_immediately(self):
        tracker = ThresholdTracker()
        thresholds = Thresholds(warning_at=3, critical_at=5)
        assert tracker.update("plaza", 1, thresholds) == ThresholdLevel.OK
        assert tracker.update("plaza", 3, thresholds) == ThresholdLevel.WARNING
        assert tracker.update("plaza", 9, thresholds) == ThresholdLevel.CRITICAL

    def test_hysteresis_holds_level_at_boundary(self):
        tracker = ThresholdTracker(hysteresis=1)
        thresholds = Thresholds(warning_at=3, critical_at=5)
        tracker.update("plaza", 5, thresholds)  # -> CRITICAL
        # 4 is below critical_at but within hysteresis: stays CRITICAL.
        assert tracker.update("plaza", 4, thresholds) == ThresholdLevel.CRITICAL
        # 3 drops to WARNING, then holds there (within hysteresis of warning_at).
        assert tracker.update("plaza", 3, thresholds) == ThresholdLevel.WARNING
        assert tracker.update("plaza", 2, thresholds) == ThresholdLevel.WARNING
        # 1 clears fully.
        assert tracker.update("plaza", 1, thresholds) == ThresholdLevel.OK

    def test_keys_are_independent(self):
        tracker = ThresholdTracker()
        thresholds = Thresholds(warning_at=3, critical_at=5)
        tracker.update("plaza", 5, thresholds)
        assert tracker.update("lobby", 1, thresholds) == ThresholdLevel.OK
        assert tracker.level("plaza") == ThresholdLevel.CRITICAL

    def test_reset(self):
        tracker = ThresholdTracker()
        thresholds = Thresholds(warning_at=3, critical_at=5)
        tracker.update("plaza", 5, thresholds)
        tracker.reset("plaza")
        assert tracker.level("plaza") == ThresholdLevel.OK

    def test_invalid_hysteresis(self):
        with pytest.raises(ValueError, match="hysteresis"):
            ThresholdTracker(hysteresis=-1)


class TestDisplay:
    def _status(self, level, people=6):
        return RegionStatus(region="plaza", people_count=people, group_count=2, largest_group=4, level=level)

    def test_renders_rows_with_counts_and_levels(self):
        text = render_display([self._status(ThresholdLevel.CRITICAL)])
        assert "plaza" in text
        assert "CRITICAL" in text
        assert "PEOPLE GROUP MONITOR" in text
        # counts appear in the row
        assert "       6" in text and "       2" in text

    def test_worst_level_summarised(self):
        statuses = [
            RegionStatus("quiet", 1, 1, 1, ThresholdLevel.OK),
            self._status(ThresholdLevel.WARNING),
        ]
        text = render_display(statuses)
        assert "status=WARNING" in text
        assert "TOTAL people=7" in text

    def test_empty_statuses(self):
        text = render_display([])
        assert "(no monitored regions)" in text
        assert "status=OK" in text
