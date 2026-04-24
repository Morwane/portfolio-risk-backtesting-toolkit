"""Tests for RunManifest and WarningsCollector."""

import json
import tempfile
from pathlib import Path

from src.reporting.manifest import RunManifest, WarningsCollector


class TestRunManifest:

    def test_set_mode_demo(self):
        m = RunManifest()
        m.set_mode(True)
        assert m.mode == "DEMO"

    def test_set_mode_live(self):
        m = RunManifest()
        m.set_mode(False)
        assert m.mode == "LIVE"

    def test_to_dict_keys(self):
        m = RunManifest()
        m.set_mode(True)
        m.set_universe(["a", "b"], 22)
        d = m.to_dict()
        assert "run_id" in d
        assert "mode" in d
        assert "universe" in d
        assert "settings" in d
        assert "runtime_seconds" in d

    def test_save_creates_json(self):
        m = RunManifest()
        m.set_mode(True)
        m.set_universe(["a"], 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            m.save(path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["mode"] == "DEMO"

    def test_ric_entry(self):
        m = RunManifest()
        m.add_ric_entry("us_large_cap", "SPY", "TR.PriceClose", "primary_ok", 0)
        d = m.to_dict()
        assert "us_large_cap" in d["ric_mapping"]
        assert d["ric_mapping"]["us_large_cap"]["ric"] == "SPY"

    def test_integrity_status(self):
        m = RunManifest()
        m.set_integrity_status("strategic_diversified", "VALID")
        d = m.to_dict()
        assert d["integrity_status"]["strategic_diversified"] == "VALID"


class TestWarningsCollector:

    def test_empty_on_init(self):
        w = WarningsCollector()
        assert w.count() == 0

    def test_add_increments_count(self):
        w = WarningsCollector()
        w.add("DATA_QUALITY", "Test warning")
        assert w.count() == 1

    def test_category_filter(self):
        w = WarningsCollector()
        w.add("DATA_QUALITY", "dq warning")
        w.add("PORTFOLIO", "portfolio warning")
        dq = w.get_by_category("DATA_QUALITY")
        assert len(dq) == 1
        assert dq[0]["message"] == "dq warning"

    def test_save_creates_json(self):
        w = WarningsCollector()
        w.add("DATA_QUALITY", "Test warning", {"detail": "extra"})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "warnings.json"
            w.save(path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["total_warnings"] == 1
            assert data["warnings"][0]["category"] == "DATA_QUALITY"

    def test_warnings_by_category_in_output(self):
        w = WarningsCollector()
        w.add("A", "msg1")
        w.add("A", "msg2")
        w.add("B", "msg3")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "warnings.json"
            w.save(path)
            data = json.loads(path.read_text())
            assert data["warnings_by_category"]["A"] == 2
            assert data["warnings_by_category"]["B"] == 1
