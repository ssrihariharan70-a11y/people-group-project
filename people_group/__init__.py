"""People Group — YOLO people detection with group-in-region and crowd alerting."""

from .alerts import (
    AlertDeliveryError,
    AlertTransport,
    CrowdAlertConnector,
    HttpTransport,
    InMemoryTransport,
    RecordedRequest,
)
from .app import FrameResult, PeopleGroupPipeline
from .detection import (
    Frame,
    PeopleDetector,
    ScriptedDetector,
    YoloPeopleDetector,
    person_at,
)
from .events import Event, EventLog, EventType
from .grouping import assign_regions, cluster_people, find_groups, footprint_distance
from .models import BBox, Group, PersonDetection, Point, Region, ThresholdLevel
from .rules import GroupRule, RuleStore
from .thresholds import RegionStatus, Thresholds, ThresholdTracker, render_display

__version__ = "0.1.0"

__all__ = [
    "AlertDeliveryError",
    "AlertTransport",
    "BBox",
    "CrowdAlertConnector",
    "Event",
    "EventLog",
    "EventType",
    "Frame",
    "FrameResult",
    "Group",
    "GroupRule",
    "HttpTransport",
    "InMemoryTransport",
    "PeopleDetector",
    "PeopleGroupPipeline",
    "PersonDetection",
    "Point",
    "RecordedRequest",
    "Region",
    "RegionStatus",
    "RuleStore",
    "ScriptedDetector",
    "ThresholdLevel",
    "ThresholdTracker",
    "Thresholds",
    "YoloPeopleDetector",
    "assign_regions",
    "cluster_people",
    "find_groups",
    "footprint_distance",
    "person_at",
    "render_display",
]
