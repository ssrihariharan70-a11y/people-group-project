"""End-to-end pipeline tests: detection -> groups -> thresholds -> events -> alerts."""

import pytest

from people_group.alerts import CrowdAlertConnector, InMemoryTransport
from people_group.app import PeopleGroupPipeline
from people_group.detection import ScriptedDetector, person_at
from people_group.events import EventType
from people_group.models import Region, ThresholdLevel
from people_group.rules import GroupRule


def make_pipeline(with_connector: bool = False):
    detector = ScriptedDetector()
    transport = InMemoryTransport() if with_connector else None
    connector = (
        CrowdAlertConnector("https://platform.example.com/api/crowd", transport=transport)
        if with_connector
        else None
    )
    pipeline = PeopleGroupPipeline(detector, connector=connector)
    pipeline.add_region(Region.rectangle("plaza", 0, 0, 600, 600))
    pipeline.rules.add(GroupRule(region="plaza", warning_at=3, critical_at=5))
    return pipeline, detector, transport


ALONE = [person_at(100, 100)]
TRIO = [person_at(100, 100), person_at(140, 100), person_at(180, 100)]
CROWD = TRIO + [person_at(90, 150), person_at(230, 140), person_at(300, 160)]


class TestPipelineFlow:
    def test_lone_person_is_not_a_group(self):
        pipeline, detector, _ = make_pipeline()
        detector.add_scene(0, ALONE)
        result = pipeline.process_frame(0)
        assert result.person_count == 1
        assert result.groups == []
        assert result.statuses[0].people_count == 0
        assert result.statuses[0].level == ThresholdLevel.OK
        assert result.events == []

    def test_group_formation_emits_event_and_warning(self):
        pipeline, detector, _ = make_pipeline()
        detector.add_scene(0, ALONE)
        detector.add_scene(1, TRIO)
        pipeline.process_frame(0)

        result = pipeline.process_frame(1)
        status = result.statuses[0]
        assert (status.people_count, status.group_count, status.largest_group) == (3, 1, 3)
        assert status.level == ThresholdLevel.WARNING
        types = [e.type for e in result.events]
        assert types == [EventType.GROUP_FORMED, EventType.THRESHOLD_WARNING]

    def test_growth_to_critical_sends_crowd_alert(self):
        pipeline, detector, transport = make_pipeline(with_connector=True)
        detector.add_scene(0, TRIO)
        detector.add_scene(1, CROWD)
        pipeline.process_frame(0)

        result = pipeline.process_frame(1)
        assert result.statuses[0].level == ThresholdLevel.CRITICAL
        types = [e.type for e in result.events]
        assert EventType.THRESHOLD_CRITICAL in types
        assert EventType.CROWD_ALERT in types

        # the alert was actually posted to the platform transport
        assert len(transport.requests) == 1
        assert transport.requests[0].json()["event"]["people_count"] == 6

        alert = result.events[-1]
        assert alert.payload["delivered"] is True
        assert alert.payload["endpoint"] == "https://platform.example.com/api/crowd"

    def test_crowd_clearing_emits_dissolved_and_cleared(self):
        pipeline, detector, _ = make_pipeline()
        detector.add_scene(0, CROWD)
        detector.add_scene(1, [])
        pipeline.process_frame(0)

        result = pipeline.process_frame(1)
        types = [e.type for e in result.events]
        assert EventType.GROUP_DISSOLVED in types
        assert EventType.THRESHOLD_CLEARED in types
        assert result.statuses[0].level == ThresholdLevel.OK

    def test_stable_group_does_not_reemit_formed(self):
        pipeline, detector, _ = make_pipeline()
        detector.add_scene(0, TRIO)
        detector.add_scene(1, TRIO)
        pipeline.process_frame(0)

        result = pipeline.process_frame(1)
        assert [e.type for e in result.events] == []  # same group, same level

    def test_alert_not_sent_without_connector(self):
        pipeline, detector, _ = make_pipeline(with_connector=False)
        detector.add_scene(0, CROWD)
        result = pipeline.process_frame(0)
        alert = [e for e in result.events if e.type == EventType.CROWD_ALERT][0]
        assert alert.payload == {"delivered": False, "endpoint": None}

    def test_display_available_on_result(self):
        pipeline, detector, _ = make_pipeline()
        detector.add_scene(0, TRIO)
        display = pipeline.process_frame(0).display
        assert "PEOPLE GROUP MONITOR" in display
        assert "plaza" in display
        assert "WARNING" in display


class TestGlobalRules:
    def test_rule_without_geometry_counts_every_region(self):
        detector = ScriptedDetector()
        pipeline = PeopleGroupPipeline(detector)
        pipeline.add_region(Region.rectangle("east", 0, 0, 300, 300))
        pipeline.add_region(Region.rectangle("west", 400, 0, 700, 300))
        pipeline.rules.add(GroupRule(region="site-wide", warning_at=2, critical_at=4))

        # pairs in both regions; singletons would be filtered by min_group_size
        detector.add_scene(0, [person_at(100, 100), person_at(140, 100), person_at(500, 100), person_at(540, 100)])
        result = pipeline.process_frame(0)
        assert len(result.groups) == 2
        assert {g.region for g in result.groups} == {"east", "west"}
        assert result.statuses[0].region == "site-wide"
        assert result.statuses[0].people_count == 4
        assert result.statuses[0].group_count == 2

    def test_disabled_rules_are_ignored(self):
        pipeline, detector, _ = make_pipeline()
        pipeline.rules.add(GroupRule(region="plaza", enabled=False), overwrite=True)
        detector.add_scene(0, CROWD)
        assert pipeline.process_frame(0).statuses == []


class TestFrameIds:
    def test_auto_incrementing_frame_ids(self):
        pipeline, detector, _ = make_pipeline()
        detector.add_scene(0, TRIO)
        detector.add_scene(1, TRIO)
        assert pipeline.process_frame(0).frame_id == 0
        assert pipeline.process_frame(1).frame_id == 1
        assert pipeline.process_frame(2).frame_id == 2

    def test_events_carry_frame_id(self):
        pipeline, detector, _ = make_pipeline()
        detector.add_scene(7, TRIO)
        result = pipeline.process_frame(7)
        assert all(e.frame_id == 7 for e in result.events)
