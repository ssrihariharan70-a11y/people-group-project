"""Tests for rule validation and JSON storage (task 3)."""

import json

import pytest

from people_group.models import ThresholdLevel
from people_group.rules import GroupRule, RuleStore


class TestGroupRule:
    def test_thresholds_property(self):
        rule = GroupRule(region="plaza", warning_at=2, critical_at=4)
        assert rule.thresholds.evaluate(2) == ThresholdLevel.WARNING
        assert rule.thresholds.evaluate(4) == ThresholdLevel.CRITICAL

    def test_validation(self):
        with pytest.raises(ValueError, match="region"):
            GroupRule(region=" ")
        with pytest.raises(ValueError, match="min_group_size"):
            GroupRule(region="plaza", min_group_size=1)
        with pytest.raises(ValueError, match="warning_at"):
            GroupRule(region="plaza", warning_at=0)
        with pytest.raises(ValueError, match="critical_at"):
            GroupRule(region="plaza", warning_at=5, critical_at=3)

    def test_round_trip_dict(self):
        rule = GroupRule(region="plaza", min_group_size=3, warning_at=4, critical_at=6, enabled=False)
        assert GroupRule.from_dict(rule.to_dict()) == rule

    def test_from_dict_defaults(self):
        rule = GroupRule.from_dict({"region": "lobby"})
        assert (rule.min_group_size, rule.warning_at, rule.critical_at, rule.enabled) == (2, 3, 5, True)


class TestRuleStore:
    def test_add_get_remove(self):
        store = RuleStore()
        rule = GroupRule(region="plaza")
        store.add(rule)
        assert store.get("plaza") == rule
        assert "plaza" in store
        assert len(store) == 1
        assert store.regions() == ["plaza"]
        removed = store.remove("plaza")
        assert removed == rule
        assert store.get("plaza") is None

    def test_add_duplicate_raises(self):
        store = RuleStore()
        store.add(GroupRule(region="plaza"))
        with pytest.raises(KeyError, match="already exists"):
            store.add(GroupRule(region="plaza", warning_at=4, critical_at=9))
        # overwrite=True replaces silently
        store.add(GroupRule(region="plaza", warning_at=4, critical_at=9), overwrite=True)
        assert store.get("plaza").warning_at == 4

    def test_update_requires_existing(self):
        store = RuleStore()
        with pytest.raises(KeyError, match="no rule"):
            store.update(GroupRule(region="ghost"))
        store.add(GroupRule(region="ghost", warning_at=3))
        store.update(GroupRule(region="ghost", warning_at=4, critical_at=7))
        assert store.get("ghost").warning_at == 4

    def test_remove_missing_raises(self):
        with pytest.raises(KeyError, match="no rule"):
            RuleStore().remove("ghost")

    def test_enabled_filter(self):
        store = RuleStore()
        store.add(GroupRule(region="a"))
        store.add(GroupRule(region="b", enabled=False))
        assert [r.region for r in store.enabled()] == ["a"]
        assert len(store.all()) == 2

    def test_missing_file_starts_empty(self, tmp_path):
        store = RuleStore(tmp_path / "rules.json")
        assert len(store) == 0

    def test_persistence_round_trip(self, tmp_path):
        path = tmp_path / "rules.json"
        store = RuleStore(path)
        store.add(GroupRule(region="plaza", warning_at=3, critical_at=5))
        store.add(GroupRule(region="lobby", min_group_size=4))
        assert path.exists()

        reloaded = RuleStore(path)
        assert len(reloaded) == 2
        assert reloaded.get("plaza").critical_at == 5
        assert reloaded.get("lobby").min_group_size == 4

        # removals persist too
        reloaded.remove("lobby")
        assert len(RuleStore(path)) == 1

    def test_file_is_valid_json_with_version(self, tmp_path):
        path = tmp_path / "rules.json"
        store = RuleStore(path)
        store.add(GroupRule(region="plaza"))
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["version"] == RuleStore.VERSION
        assert data["rules"][0]["region"] == "plaza"

    def test_unsupported_version_raises(self, tmp_path):
        path = tmp_path / "rules.json"
        path.write_text(json.dumps({"version": 999, "rules": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="version"):
            RuleStore(path)

    def test_in_memory_store_never_writes(self, tmp_path):
        store = RuleStore(tmp_path / "rules.json", auto_save=False)
        store.add(GroupRule(region="plaza"))
        store.save()  # explicit save still works
        assert RuleStore(tmp_path / "rules.json").get("plaza") is not None
