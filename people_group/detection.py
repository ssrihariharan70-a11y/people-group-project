"""People detection backends: YOLO (task 1) and a scripted fallback for tests/demos."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .models import BBox, PersonDetection

Frame = Any


@runtime_checkable
class PeopleDetector(Protocol):
    """Anything that can detect people in a frame."""

    def detect(self, frame: Frame) -> list[PersonDetection]:
        ...


class YoloPeopleDetector:
    """People detection with a YOLO model via the optional `ultralytics` package.

    Only the COCO 'person' class (id 0) is returned, filtered by confidence.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.4,
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO  # noqa: PLC0415 - optional dependency
        except ImportError as exc:
            raise RuntimeError(
                "YoloPeopleDetector requires the optional 'ultralytics' package. "
                "Install it with: pip install ultralytics opencv-python"
            ) from exc
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be within [0, 1], got {confidence}")
        self._model = YOLO(model_name)
        self._confidence = confidence
        self._device = device

    def detect(self, frame: Frame) -> list[PersonDetection]:
        results = self._model.predict(
            frame,
            conf=self._confidence,
            device=self._device,
            classes=[PersonDetection.PERSON_CLASS_ID],
            verbose=False,
        )
        return self._to_detections(results[0])

    @staticmethod
    def _to_detections(result: Any) -> list[PersonDetection]:
        detections: list[PersonDetection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections
        for box in boxes:
            class_id = int(box.cls[0])
            if class_id != PersonDetection.PERSON_CLASS_ID:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append(
                PersonDetection(BBox(x1, y1, x2, y2), confidence=float(box.conf[0]), class_id=class_id)
            )
        return detections


def person_at(
    x: float,
    y: float,
    *,
    width: float = 40.0,
    height: float = 80.0,
    confidence: float = 0.9,
) -> PersonDetection:
    """Build a person box whose footprint (bottom-center) sits at (x, y)."""
    return PersonDetection(BBox(x - width / 2, y - height, x + width / 2, y), confidence=confidence)


class ScriptedDetector:
    """Deterministic detector keyed by frame id — used by tests and the demo.

    ``detect(frame)`` treats ``frame`` as an integer frame id and returns the
    scripted detections for that frame, falling back to ``default``.
    """

    def __init__(
        self,
        scenes: Mapping[int, Sequence[PersonDetection]] | None = None,
        default: Sequence[PersonDetection] = (),
    ) -> None:
        self._scenes: dict[int, list[PersonDetection]] = {int(k): list(v) for k, v in (scenes or {}).items()}
        self._default = list(default)
        self.calls: list[int] = []

    def add_scene(self, frame_id: int, detections: Sequence[PersonDetection]) -> None:
        self._scenes[int(frame_id)] = list(detections)

    def detect(self, frame: Frame) -> list[PersonDetection]:
        frame_id = int(frame)
        self.calls.append(frame_id)
        return list(self._scenes.get(frame_id, self._default))
