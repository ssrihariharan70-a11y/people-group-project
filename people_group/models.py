"""Core data models for the People Group project."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Sequence, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"invalid bounding box: {self!r}")

    @property
    def center(self) -> Point:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def footprint(self) -> Point:
        """Bottom-center point — a robust proxy for where a person stands."""
        return ((self.x1 + self.x2) / 2.0, float(self.y2))

    def union(self, other: "BBox") -> "BBox":
        return BBox(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            max(self.x2, other.x2),
            max(self.y2, other.y2),
        )

    @staticmethod
    def union_all(boxes: Sequence["BBox"]) -> "BBox":
        if not boxes:
            raise ValueError("cannot union an empty sequence of boxes")
        result = boxes[0]
        for box in boxes[1:]:
            result = result.union(box)
        return result


@dataclass(frozen=True)
class PersonDetection:
    """A single detected person (YOLO 'person' class)."""

    PERSON_CLASS_ID: ClassVar[int] = 0

    bbox: BBox
    confidence: float
    class_id: int = PERSON_CLASS_ID
    track_id: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be within [0, 1], got {self.confidence}")

    @property
    def label(self) -> str:
        return "person"


class ThresholdLevel(str, Enum):
    """Severity of a group count relative to its thresholds."""

    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _LEVEL_RANK[self]


_LEVEL_RANK = {
    ThresholdLevel.OK: 0,
    ThresholdLevel.WARNING: 1,
    ThresholdLevel.CRITICAL: 2,
}


@dataclass(frozen=True)
class Region:
    """A watched area, defined as a polygon (4 points for a rectangle)."""

    name: str
    polygon: Tuple[Point, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("region name must not be empty")
        points = tuple((float(x), float(y)) for x, y in self.polygon)
        if len(points) < 3:
            raise ValueError(f"region '{self.name}' needs at least 3 polygon points, got {len(points)}")
        object.__setattr__(self, "polygon", points)

    @staticmethod
    def rectangle(name: str, x1: float, y1: float, x2: float, y2: float) -> "Region":
        if x2 <= x1 or y2 <= y1:
            raise ValueError("rectangle must have positive width and height")
        return Region(name=name, polygon=((x1, y1), (x2, y1), (x2, y2), (x1, y2)))

    def contains(self, point: Point) -> bool:
        """Ray-casting point-in-polygon test."""
        x, y = point
        inside = False
        count = len(self.polygon)
        j = count - 1
        for i in range(count):
            xi, yi = self.polygon[i]
            xj, yj = self.polygon[j]
            if (yi > y) != (yj > y):
                x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < x_cross:
                    inside = not inside
            j = i
        return inside


@dataclass
class Group:
    """A cluster of people counted as one group, optionally bound to a region."""

    id: int
    detections: Tuple[PersonDetection, ...]
    region: str | None = None

    def __post_init__(self) -> None:
        if not self.detections:
            raise ValueError("a group must contain at least one detection")
        self.detections = tuple(self.detections)

    @property
    def size(self) -> int:
        return len(self.detections)

    @property
    def bbox(self) -> BBox:
        return BBox.union_all([d.bbox for d in self.detections])

    @property
    def centroid(self) -> Point:
        """Average footprint of the members (where the group stands)."""
        points = [d.bbox.footprint for d in self.detections]
        return (
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points),
        )

    @property
    def mean_confidence(self) -> float:
        return sum(d.confidence for d in self.detections) / len(self.detections)
