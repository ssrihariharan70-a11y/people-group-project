"""Tests for group-in-region logic (task 1)."""

import pytest

from people_group.detection import person_at
from people_group.grouping import assign_regions, cluster_people, find_groups
from people_group.models import BBox, Group, PersonDetection, Region


class TestClustering:
    def test_nearby_people_form_one_group(self):
        # Chain of 40px steps links under a 90px threshold.
        detections = [person_at(0, 0), person_at(40, 0), person_at(80, 0)]
        clusters = cluster_people(detections, max_link_distance=90)
        assert clusters == [[0, 1, 2]]

    def test_far_apart_people_stay_separate(self):
        detections = [person_at(0, 0), person_at(500, 0)]
        clusters = cluster_people(detections, max_link_distance=90)
        assert clusters == [[0], [1]]

    def test_chain_linking_merges_cluster(self):
        # 0-1 and 1-2 are close, but 0 and 2 are far: single linkage still merges.
        detections = [person_at(0, 0), person_at(80, 0), person_at(160, 0)]
        clusters = cluster_people(detections, max_link_distance=90)
        assert clusters == [[0, 1, 2]]

    def test_empty_input(self):
        assert cluster_people([]) == []

    def test_invalid_distance_raises(self):
        with pytest.raises(ValueError, match="positive"):
            cluster_people([person_at(0, 0)], max_link_distance=0)


class TestGroupFormation:
    def test_single_person_is_not_a_group_by_default(self):
        groups = find_groups([person_at(0, 0)], max_link_distance=90)
        assert groups == []

    def test_min_group_size_filters_clusters(self):
        detections = [person_at(0, 0), person_at(40, 0), person_at(800, 0)]
        assert len(find_groups(detections, min_group_size=2)) == 1
        assert find_groups(detections, min_group_size=3) == []
        assert len(find_groups(detections, min_group_size=1)) == 2

    def test_group_geometry(self):
        group = find_groups([person_at(0, 0), person_at(80, 0)])[0]
        assert group.size == 2
        assert group.centroid == pytest.approx((40, 0))
        # Each person box is 40 wide: spans -20..100.
        assert group.bbox == BBox(-20, -80, 100, 0)

    def test_group_requires_members(self):
        with pytest.raises(ValueError, match="at least one"):
            Group(id=1, detections=())

    def test_invalid_min_group_size_raises(self):
        with pytest.raises(ValueError, match="min_group_size"):
            find_groups([person_at(0, 0)], min_group_size=0)


class TestRegionBinding:
    def test_group_inside_region_is_bound(self):
        region = Region.rectangle("east", -50, -50, 200, 50)
        detections = [person_at(0, 0), person_at(40, 0)]
        groups = find_groups(detections, regions=[region])
        assert [g.region for g in groups] == ["east"]

    def test_group_outside_region_is_unbound(self):
        region = Region.rectangle("east", -50, -50, 200, 50)
        detections = [person_at(0, 0), person_at(900, 900)]  # far apart -> two singletons
        groups = find_groups(detections, regions=[region], min_group_size=1)
        assert [g.region for g in groups] == ["east", None]

    def test_group_not_in_region_when_members_are_a_minority(self):
        # Only 1 of 3 members is inside; vote threshold 0.5 rejects the region.
        region = Region.rectangle("booth", -50, -50, 50, 50)
        detections = [person_at(0, 0), person_at(120, 0), person_at(240, 0)]
        groups = find_groups(detections, regions=[region], max_link_distance=130)
        assert len(groups) == 1
        assert groups[0].region is None

    def test_majority_vote_binds_group(self):
        region = Region.rectangle("booth", -50, -50, 200, 50)
        detections = [person_at(0, 0), person_at(60, 0), person_at(300, 0)]
        groups = find_groups(detections, regions=[region], max_link_distance=310)
        assert groups[0].region == "booth"

    def test_polygon_region(self):
        triangle = Region(name="diagonal", polygon=((0, 0), (200, 0), (0, 200)))
        assert triangle.contains((30, 30))
        assert not triangle.contains((180, 180))

    def test_region_validation(self):
        with pytest.raises(ValueError, match="empty"):
            Region(name="  ", polygon=((0, 0), (1, 0), (0, 1)))
        with pytest.raises(ValueError, match="3 polygon points"):
            Region(name="line", polygon=((0, 0), (1, 1)))
        with pytest.raises(ValueError, match="positive"):
            Region.rectangle("bad", 10, 10, 0, 50)

    def test_assign_regions_invalid_vote_threshold(self):
        detections = [person_at(0, 0)]
        with pytest.raises(ValueError, match="region_vote_threshold"):
            assign_regions(detections, [[0]], region_vote_threshold=0)
