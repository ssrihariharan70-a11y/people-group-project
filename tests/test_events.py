"""Tests for group events (task 3)."""

import json

import pytest

from people_group.events import Event, EventLog, EventType
from people_group.models import ThresholdLevel


class TestEvent:
    def test_to_dict_is_json_serialisable(self):
        event = Event(
            type=EventType.THRESHOLD_CRITICAL,
            region="plaza",
            message="crowd building",
            people_count=6,
            group_count=2,
            level=ThresholdLevel.CRITICAL,
            frame_id=4,
            payload={"delivered": True},
        )
        data = json.loads(json.dumps(event.to_dict()))
        assert data["type"] == "threshold_critical"
        assert data["level"] == "critical"
        assert data["payload"] == {"delivered": True}
        assert data["timestamp"] > 0

    def test_optional_fields_default_to_none(self):
        data = Event(type=EventType.GROUP_FORMED).to_dict()
        assert data["region"] is None
        assert data["level"] is None


class TestEventLog:
    def test_emit_records_and_dispatches(self):
        log = EventLog()
        seen = []
        log.subscribe(seen.append)
        event = Event(type=EventType.GROUP_FORMED, region="plaza")
        assert log.emit(event) is event
        assert seen == [event]
        assert list(log) == [event]
        assert len(log) == 1

    def test_multiple_subscribers_and_unsubscribe(self):
        log = EventLog()
        first, second = [], []
        handler_a, handler_b = first.append, second.append
        log.subscribe(handler_a)
        log.subscribe(handler_b)
        log.emit(Event(type=EventType.GROUP_FORMED))
        log.unsubscribe(handler_a)
        log.emit(Event(type=EventType.GROUP_DISSOLVED))
        assert len(first) == 1
        assert len(second) == 2

    def test_failing_handler_does_not_break_emit(self):
        log = EventLog()
        good = []

        def bad_handler(event):
            raise RuntimeError("boom")

        log.subscribe(bad_handler)
        log.subscribe(good.append)
        event = log.emit(Event(type=EventType.CROWD_ALERT))
        assert list(log) == [event]  # recorded despite the handler failure
        assert good == [event]
        assert len(log.errors) == 1
        assert "boom" in str(log.errors[0])

    def test_recent_capacity(self):
        log = EventLog(capacity=3)
        for i in range(5):
            log.emit(Event(type=EventType.GROUP_FORMED, frame_id=i))
        assert [e.frame_id for e in log.recent(10)] == [2, 3, 4]
        assert len(log) == 3

    def test_of_type_filter(self):
        log = EventLog()
        log.emit(Event(type=EventType.GROUP_FORMED))
        log.emit(Event(type=EventType.THRESHOLD_WARNING))
        log.emit(Event(type=EventType.GROUP_FORMED))
        assert len(log.of_type(EventType.GROUP_FORMED)) == 2
        assert len(log.of_type(EventType.THRESHOLD_WARNING)) == 1

    def test_clear(self):
        log = EventLog()
        log.emit(Event(type=EventType.GROUP_FORMED))
        log.clear()
        assert len(log) == 0

    def test_validation(self):
        with pytest.raises(ValueError, match="capacity"):
            EventLog(capacity=0)
        with pytest.raises(ValueError, match="n must be"):
            EventLog().recent(-1)
