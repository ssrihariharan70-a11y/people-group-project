"""The People Group pipeline: detects, groups, evaluates, emits and alerts.

One call to :meth:`PeopleGroupPipeline.process_frame` runs the whole chain
for a single frame:

1. YOLO people detection (any :class:`~people_group.detection.PeopleDetector`)
2. group-in-region logic (:mod:`people_group.grouping`)
3. per-rule count + threshold evaluation with hysteresis
4. events: group formed/dissolved, threshold changes
5. crowd alerts to the platform when a rule goes critical
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .alerts import CrowdAlertConnector
from .detection import PeopleDetector
from .events import Event, EventLog, EventType
from .grouping import find_groups
from .models import Group, PersonDetection, Region, ThresholdLevel
from .rules import GroupRule, RuleStore
from .thresholds import RegionStatus, ThresholdTracker, render_display


@dataclass
class FrameResult:
    """Everything the pipeline decided for one frame."""

    frame_id: int
    person_count: int
    groups: List[Group]
    statuses: List[RegionStatus]
    events: List[Event] = field(default_factory=list)

    @property
    def display(self) -> str:
        return render_display(self.statuses)

    @property
    def worst_level(self) -> ThresholdLevel:
        return max((s.level for s in self.statuses), key=lambda level: level.rank, default=ThresholdLevel.OK)


def _diff_groups(
    previous: Sequence[Group],
    current: Sequence[Group],
    max_distance: float,
) -> tuple[List[Group], List[Group]]:
    """Match previous to current groups by centroid proximity.

    Returns ``(new_groups, dissolved_groups)``.
    """
    pairs = sorted(
        (
            (math.hypot(prev.centroid[0] - cur.centroid[0], prev.centroid[1] - cur.centroid[1]), pi, ci)
            for pi, prev in enumerate(previous)
            for ci, cur in enumerate(current)
        ),
    )
    matched_prev: set[int] = set()
    matched_current: set[int] = set()
    for distance, pi, ci in pairs:
        if distance <= max_distance and pi not in matched_prev and ci not in matched_current:
            matched_prev.add(pi)
            matched_current.add(ci)
    new_groups = [group for ci, group in enumerate(current) if ci not in matched_current]
    dissolved = [group for pi, group in enumerate(previous) if pi not in matched_prev]
    return new_groups, dissolved


class PeopleGroupPipeline:
    """Stateful frame processor tying detection, grouping, rules and alerts."""

    def __init__(
        self,
        detector: PeopleDetector,
        rules: Optional[RuleStore] = None,
        connector: Optional[CrowdAlertConnector] = None,
        events: Optional[EventLog] = None,
        *,
        max_link_distance: float = 90.0,
        region_vote_threshold: float = 0.5,
        group_match_distance: float = 90.0,
        hysteresis: int = 1,
    ) -> None:
        self._detector = detector
        self._rules = rules if rules is not None else RuleStore()
        self._connector = connector
        self._events = events if events is not None else EventLog()
        self._regions: Dict[str, Region] = {}
        self._tracker = ThresholdTracker(hysteresis=hysteresis)
        self._max_link_distance = max_link_distance
        self._region_vote_threshold = region_vote_threshold
        self._group_match_distance = group_match_distance
        self._previous_groups: Dict[str, List[Group]] = {}
        self._next_frame_id = 0

    # -- wiring ---------------------------------------------------------
    @property
    def detector(self) -> PeopleDetector:
        return self._detector

    @detector.setter
    def detector(self, detector: PeopleDetector) -> None:
        self._detector = detector

    @property
    def rules(self) -> RuleStore:
        return self._rules

    @property
    def connector(self) -> Optional[CrowdAlertConnector]:
        return self._connector

    @connector.setter
    def connector(self, connector: Optional[CrowdAlertConnector]) -> None:
        self._connector = connector

    @property
    def events(self) -> EventLog:
        return self._events

    @property
    def regions(self) -> Dict[str, Region]:
        return dict(self._regions)

    def add_region(self, region: Region) -> Region:
        self._regions[region.name] = region
        return region

    def subscribe(self, handler: Callable[[Event], None]) -> None:
        self._events.subscribe(handler)

    # -- frame processing ------------------------------------------------
    def process_frame(self, frame=None, frame_id: Optional[int] = None) -> FrameResult:
        """Run the full chain for one frame; ids auto-increment unless given.

        A plain ``int`` frame is treated as the frame id itself (used by the
        scripted detector, the demo, and tests).
        """
        if frame_id is None:
            # An int frame doubles as its id (handy for scripted/demo frames).
            frame_id = frame if isinstance(frame, int) else self._next_frame_id
        self._next_frame_id = max(self._next_frame_id, frame_id) + 1

        detections = self._detector.detect(frame)
        groups = find_groups(
            detections,
            regions=list(self._regions.values()),
            max_link_distance=self._max_link_distance,
            region_vote_threshold=self._region_vote_threshold,
        )

        emitted: List[Event] = []
        emitted.extend(self._emit_group_lifecycle(groups, frame_id))

        statuses: List[RegionStatus] = []
        for rule in self._rules.enabled():
            status = self._evaluate_rule(rule, groups, frame_id, emitted)
            statuses.append(status)

        self._maybe_send_crowd_alerts(statuses, frame_id, emitted)
        self._previous_groups = self._group_by_key(groups)

        return FrameResult(
            frame_id=frame_id,
            person_count=len(detections),
            groups=groups,
            statuses=statuses,
            events=emitted,
        )

    # -- internals -------------------------------------------------------
    @staticmethod
    def _key(group: Group) -> str:
        return group.region or ""

    def _group_by_key(self, groups: Sequence[Group]) -> Dict[str, List[Group]]:
        by_key: Dict[str, List[Group]] = {}
        for group in groups:
            by_key.setdefault(self._key(group), []).append(group)
        return by_key

    def _emit_group_lifecycle(self, groups: Sequence[Group], frame_id: int) -> List[Event]:
        emitted: List[Event] = []
        current_by_key = self._group_by_key(groups)
        keys = set(self._previous_groups) | set(current_by_key)
        for key in keys:
            new_groups, dissolved = _diff_groups(
                self._previous_groups.get(key, []),
                current_by_key.get(key, []),
                self._group_match_distance,
            )
            label = key or "unassigned"
            for group in new_groups:
                emitted.append(
                    self._events.emit(
                        Event(
                            type=EventType.GROUP_FORMED,
                            region=group.region,
                            message=f"Group of {group.size} formed in {label}",
                            people_count=group.size,
                            group_count=1,
                            group_id=group.id,
                            frame_id=frame_id,
                        )
                    )
                )
            for group in dissolved:
                emitted.append(
                    self._events.emit(
                        Event(
                            type=EventType.GROUP_DISSOLVED,
                            region=group.region,
                            message=f"Group of {group.size} dissolved in {label}",
                            people_count=group.size,
                            group_count=1,
                            group_id=group.id,
                            frame_id=frame_id,
                        )
                    )
                )
        return emitted

    def _evaluate_rule(self, rule: GroupRule, groups: Sequence[Group], frame_id: int, emitted: List[Event]) -> RegionStatus:
        # A rule without registered geometry acts as a global (all-regions) rule.
        is_global = rule.region not in self._regions
        matching = [
            group
            for group in groups
            if group.size >= rule.min_group_size and (is_global or group.region == rule.region)
        ]
        people_count = sum(group.size for group in matching)
        largest = max((group.size for group in matching), default=0)

        previous_level = self._tracker.level(rule.region)
        level = self._tracker.update(rule.region, people_count, rule.thresholds)

        if level != previous_level:
            event_type = {
                ThresholdLevel.OK: EventType.THRESHOLD_CLEARED,
                ThresholdLevel.WARNING: EventType.THRESHOLD_WARNING,
                ThresholdLevel.CRITICAL: EventType.THRESHOLD_CRITICAL,
            }[level]
            emitted.append(
                self._events.emit(
                    Event(
                        type=event_type,
                        region=rule.region,
                        message=(
                            f"{rule.region}: {people_count} people in groups "
                            f"-> {level.value.upper()}"
                        ),
                        people_count=people_count,
                        group_count=len(matching),
                        level=level,
                        frame_id=frame_id,
                    )
                )
            )

        return RegionStatus(
            region=rule.region,
            people_count=people_count,
            group_count=len(matching),
            largest_group=largest,
            level=level,
        )

    def _maybe_send_crowd_alerts(self, statuses: Sequence[RegionStatus], frame_id: int, emitted: List[Event]) -> None:
        for status in statuses:
            if status.level != ThresholdLevel.CRITICAL:
                continue
            delivered = False
            if self._connector is not None:
                alert_event = Event(
                    type=EventType.CROWD_ALERT,
                    region=status.region,
                    message=(
                        f"Crowd alert: {status.people_count} people in "
                        f"{status.group_count} group(s) at {status.region}"
                    ),
                    people_count=status.people_count,
                    group_count=status.group_count,
                    level=status.level,
                    frame_id=frame_id,
                )
                delivered = self._connector.send_alert(alert_event)
            emitted.append(
                self._events.emit(
                    Event(
                        type=EventType.CROWD_ALERT,
                        region=status.region,
                        message=(
                            f"Crowd alert at {status.region}: {status.people_count} people "
                            f"in {status.group_count} group(s)"
                        ),
                        people_count=status.people_count,
                        group_count=status.group_count,
                        level=status.level,
                        frame_id=frame_id,
                        payload={
                            "delivered": delivered,
                            "endpoint": self._connector.endpoint if self._connector else None,
                        },
                    )
                )
            )
