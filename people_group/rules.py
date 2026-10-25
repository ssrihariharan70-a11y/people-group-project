"""Group event rules and their JSON storage (task 3).

A :class:`GroupRule` links a region name to a minimum group size and
warning/critical thresholds. :class:`RuleStore` persists rules as JSON so
configuration survives restarts.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List

from .thresholds import Thresholds


@dataclass(frozen=True)
class GroupRule:
    """When to warn/escalate for groups inside a region."""

    region: str
    min_group_size: int = 2
    warning_at: int = 3
    critical_at: int = 5
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.region.strip():
            raise ValueError("rule region must not be empty")
        if self.min_group_size < 2:
            raise ValueError(f"min_group_size must be >= 2 (a group needs 2+ people), got {self.min_group_size}")
        Thresholds(self.warning_at, self.critical_at)  # validate boundaries

    @property
    def thresholds(self) -> Thresholds:
        return Thresholds(self.warning_at, self.critical_at)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GroupRule":
        return cls(
            region=str(data["region"]),
            min_group_size=int(data.get("min_group_size", 2)),
            warning_at=int(data.get("warning_at", 3)),
            critical_at=int(data.get("critical_at", 5)),
            enabled=bool(data.get("enabled", True)),
        )


class RuleStore:
    """CRUD store for group rules, persisted to a JSON file (optional)."""

    VERSION = 1

    def __init__(self, path: str | Path | None = None, auto_save: bool = True) -> None:
        self._path = Path(path) if path is not None else None
        self._auto_save = auto_save
        self._rules: Dict[str, GroupRule] = {}
        if self._path is not None and self._path.exists():
            self.load()

    # -- persistence -----------------------------------------------------
    @property
    def path(self) -> Path | None:
        return self._path

    def load(self) -> None:
        if self._path is None:
            raise ValueError("cannot load: no path configured for this store")
        data = json.loads(self._path.read_text(encoding="utf-8"))
        version = int(data.get("version", self.VERSION))
        if version != self.VERSION:
            raise ValueError(f"unsupported rules file version {version} (expected {self.VERSION})")
        self._rules = {}
        for entry in data.get("rules", []):
            rule = GroupRule.from_dict(entry)
            self._rules[rule.region] = rule

    def save(self) -> None:
        if self._path is None:
            return
        payload = {
            "version": self.VERSION,
            "rules": [rule.to_dict() for rule in self._rules.values()],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, self._path)

    def _maybe_save(self) -> None:
        if self._auto_save:
            self.save()

    # -- CRUD --------------------------------------------------------------
    def add(self, rule: GroupRule, overwrite: bool = False) -> None:
        if rule.region in self._rules and not overwrite:
            raise KeyError(f"a rule for region '{rule.region}' already exists")
        self._rules[rule.region] = rule
        self._maybe_save()

    def update(self, rule: GroupRule) -> None:
        if rule.region not in self._rules:
            raise KeyError(f"no rule found for region '{rule.region}'")
        self._rules[rule.region] = rule
        self._maybe_save()

    def remove(self, region: str) -> GroupRule:
        if region not in self._rules:
            raise KeyError(f"no rule found for region '{region}'")
        rule = self._rules.pop(region)
        self._maybe_save()
        return rule

    def get(self, region: str) -> GroupRule | None:
        return self._rules.get(region)

    def all(self) -> List[GroupRule]:
        return list(self._rules.values())

    def enabled(self) -> List[GroupRule]:
        return [rule for rule in self._rules.values() if rule.enabled]

    def regions(self) -> List[str]:
        return list(self._rules.keys())

    def __contains__(self, region: str) -> bool:
        return region in self._rules

    def __len__(self) -> int:
        return len(self._rules)

    def __iter__(self) -> Iterator[GroupRule]:
        return iter(self._rules.values())
