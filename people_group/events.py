"""Group events: what happened, when, and where (task 3).

Events are emitted by the pipeline (group formed/dissolved, threshold
changes, crowd alerts) and recorded in an :class:`EventLog` that also fans
them out to subscribers (display, storage, alerting).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional

from .models import ThresholdLevel


class EventType(str, Enum):
    GROUP_FORMED = "group_formed"
    GROUP_DISSOLVED = "group_dissolved"
    THRESHOLD_WARNING = "threshold_warning"
    THRESHOLD_CRITICAL = "threshold_critical"
    THRESHOLD_CLEARED = "threshold_cleared"
    CROWD_ALERT = "crowd_alert"


@dataclass(frozen=True)
class Event:
    """A single happened-in-the-world fact, serialisable to a dict."""

    type: EventType
    region: Optional[str] = None
    message: str = ""
    people_count: int = 0
    group_count: int = 0
    level: Optional[ThresholdLevel] = None
    group_id: Optional[int] = None
    frame_id: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "region": self.region,
            "message": self.message,
            "people_count": self.people_count,
            "group_count": self.group_count,
            "level": self.level.value if self.level is not None else None,
            "group_id": self.group_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "payload": dict(self.payload),
        }


EventHandler = Callable[[Event], None]


class EventLog:
    """Records events and dispatches them to subscribers.

    A failing handler never breaks the caller: the error is caught and kept in
    :attr:`errors` for inspection.
    """

    def __init__(self, capacity: Optional[int] = None) -> None:
        if capacity is not None and capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._events: List[Event] = []
        self._handlers: List[EventHandler] = []
        self.errors: List[BaseException] = []

    # -- subscriptions ---------------------------------------------------
    def subscribe(self, handler: EventHandler) -> None:
        if handler not in self._handlers:
            self._handlers.append(handler)

    def unsubscribe(self, handler: EventHandler) -> None:
        if handler in self._handlers:
            self._handlers.remove(handler)

    # -- emission & queries ----------------------------------------------
    def emit(self, event: Event) -> Event:
        self._events.append(event)
        if self._capacity is not None and len(self._events) > self._capacity:
            del self._events[: len(self._events) - self._capacity]
        for handler in list(self._handlers):
            try:
                handler(event)
            except BaseException as exc:  # noqa: BLE001 - keep the pipeline alive
                self.errors.append(exc)
        return event

    def recent(self, n: int = 10) -> List[Event]:
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        return self._events[-n:]

    def of_type(self, event_type: EventType) -> List[Event]:
        return [event for event in self._events if event.type == event_type]

    def clear(self) -> None:
        self._events.clear()

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)
