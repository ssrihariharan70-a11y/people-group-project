"""Tests for the detection backends (task 1)."""

import importlib.util

import pytest

from people_group.detection import ScriptedDetector, YoloPeopleDetector, person_at
from people_group.models import BBox, PersonDetection


class TestPersonHelper:
    def test_person_at_footprint(self):
        person = person_at(100, 200)
        assert person.bbox.footprint == (100, 200)
        assert person.label == "person"
        assert person.class_id == PersonDetection.PERSON_CLASS_ID == 0

    def test_confidence_validation(self):
        with pytest.raises(ValueError, match="confidence"):
            person_at(0, 0, confidence=1.5)


class TestScriptedDetector:
    def test_returns_scenes_by_frame_id(self):
        crowd = [person_at(0, 0), person_at(50, 0)]
        detector = ScriptedDetector({2: crowd})
        assert detector.detect(2) == crowd
        assert detector.detect(2) == crowd  # repeatable
        assert detector.calls == [2, 2]

    def test_default_fallback(self):
        detector = ScriptedDetector({0: []}, default=[person_at(0, 0)])
        assert detector.detect(0) == []  # scripted empty scene wins
        assert len(detector.detect(99)) == 1  # unsripted frames fall back to default

    def test_add_scene(self):
        detector = ScriptedDetector()
        detector.add_scene(1, [person_at(10, 10)])
        # person_at places the footprint (bottom-center) at (10, 10); box is 40x80.
        assert detector.detect(1)[0].bbox == BBox(-10, -70, 30, 10)


class TestYoloDetector:
    @pytest.mark.skipif(
        importlib.util.find_spec("ultralytics") is not None,
        reason="ultralytics installed; error path not reachable",
    )
    def test_helpful_error_without_ultralytics(self):
        with pytest.raises(RuntimeError, match="ultralytics"):
            YoloPeopleDetector()

    def test_confidence_validation_happens_after_import(self):
        if importlib.util.find_spec("ultralytics") is None:
            pytest.skip("ultralytics not installed")
        with pytest.raises(ValueError, match="confidence"):  # pragma: no cover
            YoloPeopleDetector(confidence=2.0)
