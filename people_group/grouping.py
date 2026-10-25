"""Group-in-region logic: cluster detected people and bind clusters to watched regions.

Clustering is single-linkage: people whose footprints are within
``max_link_distance`` of each other (possibly through a chain of other people)
belong to the same group. A group is considered "in" a region when at least
``region_vote_threshold`` of its members stand inside that region.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

from .models import Group, PersonDetection, Point, Region

DistanceFn = Callable[[PersonDetection, PersonDetection], float]


def _euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def footprint_distance(a: PersonDetection, b: PersonDetection) -> float:
    """Distance between where two people stand (bottom-center of their boxes)."""
    return _euclidean(a.bbox.footprint, b.bbox.footprint)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, i: int) -> int:
        while self._parent[i] != i:
            self._parent[i] = self._parent[self._parent[i]]
            i = self._parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def cluster_people(
    detections: Sequence[PersonDetection],
    max_link_distance: float = 90.0,
    distance_fn: DistanceFn = footprint_distance,
) -> list[list[int]]:
    """Single-linkage cluster people; returns clusters as sorted index lists."""
    if max_link_distance <= 0:
        raise ValueError(f"max_link_distance must be positive, got {max_link_distance}")

    union_find = _UnionFind(len(detections))
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            if distance_fn(detections[i], detections[j]) <= max_link_distance:
                union_find.union(i, j)

    clusters: list[list[int]] = []
    for i in range(len(detections)):
        if union_find.find(i) == i:
            clusters.append([k for k in range(i, len(detections)) if union_find.find(k) == i])
    return clusters


def _match_region(
    members: Sequence[PersonDetection],
    regions: Sequence[Region],
    vote_threshold: float,
) -> str | None:
    for region in regions:
        votes = sum(1 for member in members if region.contains(member.bbox.footprint))
        if members and votes / len(members) >= vote_threshold and votes > 0:
            return region.name
    return None


def assign_regions(
    detections: Sequence[PersonDetection],
    clusters: Sequence[Sequence[int]],
    regions: Sequence[Region] = (),
    region_vote_threshold: float = 0.5,
    first_group_id: int = 1,
) -> list[Group]:
    """Turn index clusters into Group objects, binding each to a region by vote."""
    if not 0.0 < region_vote_threshold <= 1.0:
        raise ValueError(f"region_vote_threshold must be within (0, 1], got {region_vote_threshold}")

    groups: list[Group] = []
    for offset, cluster in enumerate(clusters):
        members = tuple(detections[i] for i in cluster)
        groups.append(
            Group(
                id=first_group_id + offset,
                detections=members,
                region=_match_region(members, regions, region_vote_threshold),
            )
        )
    return groups


def find_groups(
    detections: Sequence[PersonDetection],
    regions: Sequence[Region] = (),
    *,
    max_link_distance: float = 90.0,
    min_group_size: int = 2,
    region_vote_threshold: float = 0.5,
) -> list[Group]:
    """Cluster people, keep clusters of at least ``min_group_size``, bind regions."""
    if min_group_size < 1:
        raise ValueError(f"min_group_size must be >= 1, got {min_group_size}")

    clusters = cluster_people(detections, max_link_distance=max_link_distance)
    clusters = [cluster for cluster in clusters if len(cluster) >= min_group_size]
    return assign_regions(
        detections,
        clusters,
        regions,
        region_vote_threshold=region_vote_threshold,
    )
