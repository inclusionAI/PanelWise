from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from draco_eval import config

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkMetadataTests(unittest.TestCase):
    def test_budget_summary_path_ignores_record_directory_trailing_slash(self) -> None:
        budget = importlib.import_module("budget_fusion_full")
        original_records_dir = budget.REC_DIR
        try:
            budget.REC_DIR = "output/run/records"
            without_slash = budget.summary_path()
            budget.REC_DIR = "output/run/records/"
            with_slash = budget.summary_path()
        finally:
            budget.REC_DIR = original_records_dir
        self.assertEqual(without_slash, "output/run/summary.json")
        self.assertEqual(with_slash, without_slash)

    def test_fusion_resume_rejects_mismatched_direct_fetch_metadata(self) -> None:
        fusion_full = importlib.import_module("fusion_full")
        original_records_dir = fusion_full.RECORDS_DIR
        original_direct_fetch = config.RESEARCH_ENABLE_DIRECT_FETCH
        config.RESEARCH_ENABLE_DIRECT_FETCH = False
        try:
            with tempfile.TemporaryDirectory() as directory:
                fusion_full.RECORDS_DIR = directory
                (Path(directory) / "task.json").write_text(json.dumps({
                    "task_id": "task", "fused": 42, "direct_fetch_enabled": True,
                }))
                task = type("Task", (), {"id": "task"})()
                with self.assertRaisesRegex(RuntimeError, "direct_fetch_enabled"):
                    asyncio.run(fusion_full.run_task(None, task, None, None))
        finally:
            fusion_full.RECORDS_DIR = original_records_dir
            config.RESEARCH_ENABLE_DIRECT_FETCH = original_direct_fetch

    def test_fusion_preflight_rejects_cache_mismatch_before_scheduling(self) -> None:
        fusion_full = importlib.import_module("fusion_full")
        original_records_dir = fusion_full.RECORDS_DIR
        original_direct_fetch = config.RESEARCH_ENABLE_DIRECT_FETCH
        config.RESEARCH_ENABLE_DIRECT_FETCH = False
        try:
            with tempfile.TemporaryDirectory() as directory:
                fusion_full.RECORDS_DIR = directory
                (Path(directory) / "task.json").write_text(json.dumps({
                    "task_id": "task", "fused": 42, "direct_fetch_enabled": True,
                }))
                task = type("Task", (), {"id": "task"})()
                with self.assertRaisesRegex(RuntimeError, "direct_fetch_enabled"):
                    fusion_full.validate_resume_cache([task])
        finally:
            fusion_full.RECORDS_DIR = original_records_dir
            config.RESEARCH_ENABLE_DIRECT_FETCH = original_direct_fetch

    def test_fusion_resume_accepts_legacy_record_only_when_direct_fetch_is_enabled(self) -> None:
        fusion_full = importlib.import_module("fusion_full")
        original_records_dir = fusion_full.RECORDS_DIR
        original_direct_fetch = config.RESEARCH_ENABLE_DIRECT_FETCH
        try:
            with tempfile.TemporaryDirectory() as directory:
                fusion_full.RECORDS_DIR = directory
                legacy = {"task_id": "task", "fused": 42}
                (Path(directory) / "task.json").write_text(json.dumps(legacy))
                task = type("Task", (), {"id": "task"})()
                config.RESEARCH_ENABLE_DIRECT_FETCH = True
                self.assertEqual(asyncio.run(fusion_full.run_task(None, task, None, None)), legacy)
                config.RESEARCH_ENABLE_DIRECT_FETCH = False
                with self.assertRaisesRegex(RuntimeError, "direct_fetch_enabled"):
                    asyncio.run(fusion_full.run_task(None, task, None, None))
        finally:
            fusion_full.RECORDS_DIR = original_records_dir
            config.RESEARCH_ENABLE_DIRECT_FETCH = original_direct_fetch

    def test_fusion_new_record_serializes_effective_direct_fetch_enabled(self) -> None:
        fusion_full = importlib.import_module("fusion_full")
        original_records_dir = fusion_full.RECORDS_DIR
        original_direct_fetch = config.RESEARCH_ENABLE_DIRECT_FETCH
        member = SimpleNamespace(
            model="model", report="report", ok=True, n_searches=0, n_fetches=0,
            n_blocked_fetches=0, n_fetch_errors=0, fetched_urls=[], blocked_fetch_urls=[],
        )
        fusion = SimpleNamespace(panel=[member], fused_report="fused report", analysis={})
        task = SimpleNamespace(id="task", domain="domain")

        async def write_record():
            return await fusion_full.run_task(
                None, task, asyncio.Semaphore(1), SimpleNamespace(cost_usd=0)
            )

        try:
            for enabled in (False, True):
                with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                    fusion_full.RECORDS_DIR = directory
                    config.RESEARCH_ENABLE_DIRECT_FETCH = enabled
                    with mock.patch.object(fusion_full, "run_fusion", new=mock.AsyncMock(return_value=fusion)), \
                         mock.patch.object(fusion_full, "score", new=mock.AsyncMock(return_value=50)):
                        result = asyncio.run(write_record())
                    serialized = json.loads((Path(directory) / "task.json").read_text())
                    self.assertEqual(result["direct_fetch_enabled"], enabled)
                    self.assertEqual(serialized["direct_fetch_enabled"], enabled)
        finally:
            fusion_full.RECORDS_DIR = original_records_dir
            config.RESEARCH_ENABLE_DIRECT_FETCH = original_direct_fetch

    def test_fusion_summary_serializes_direct_fetch_enabled(self) -> None:
        fusion_full = importlib.import_module("fusion_full")
        original_out_base = fusion_full.OUT_BASE
        original_records_dir = fusion_full.RECORDS_DIR
        original_direct_fetch = config.RESEARCH_ENABLE_DIRECT_FETCH
        task = SimpleNamespace(id="task", domain="domain")
        record = {"task_id": task.id, "domain": task.domain, "fused": 50, "solo": {}}

        class FakeClientContext:
            async def __aenter__(self):
                return object()

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

        try:
            for enabled in (False, True):
                with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                    fusion_full.OUT_BASE = directory
                    fusion_full.RECORDS_DIR = str(Path(directory) / "records")
                    config.RESEARCH_ENABLE_DIRECT_FETCH = enabled
                    with mock.patch.object(sys, "argv", ["fusion_full.py"]), \
                         mock.patch.object(fusion_full.config, "validate_provider_config"), \
                         mock.patch.object(fusion_full, "load_all", return_value=[task]), \
                         mock.patch.object(fusion_full, "make_client", return_value=FakeClientContext()), \
                         mock.patch.object(fusion_full, "run_task", new=mock.AsyncMock(return_value=record)):
                        asyncio.run(fusion_full.main())
                    serialized = json.loads((Path(directory) / "summary.json").read_text())
                    self.assertEqual(serialized["direct_fetch_enabled"], enabled)
        finally:
            fusion_full.OUT_BASE = original_out_base
            fusion_full.RECORDS_DIR = original_records_dir
            config.RESEARCH_ENABLE_DIRECT_FETCH = original_direct_fetch

    def test_budget_records_serialize_effective_direct_fetch_enabled(self) -> None:
        budget = importlib.import_module("budget_fusion_full")
        original_panel_dir = budget.PANEL_DIR
        original_records_dir = budget.REC_DIR
        original_direct_fetch = config.RESEARCH_ENABLE_DIRECT_FETCH
        task = SimpleNamespace(id="task", domain="domain")
        result = SimpleNamespace(report="report", ok=True, n_searches=0, n_fetches=0, n_steps=1)
        try:
            for enabled in (False, True):
                with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                    budget.PANEL_DIR = str(Path(directory) / "panel")
                    budget.REC_DIR = str(Path(directory) / "records")
                    config.RESEARCH_ENABLE_DIRECT_FETCH = enabled
                    with mock.patch.object(budget, "run_research_agent", new=mock.AsyncMock(return_value=result)), \
                         mock.patch.object(budget, "_analyze", new=mock.AsyncMock(return_value={})), \
                         mock.patch.object(budget, "_synthesize", new=mock.AsyncMock(return_value="fused")), \
                         mock.patch.object(budget, "grade", new=mock.AsyncMock(return_value=50)):
                        panel = asyncio.run(budget.build_panel(object(), task))
                        record = asyncio.run(
                            budget.run_task(object(), task, asyncio.Semaphore(1), SimpleNamespace(cost_usd=0))
                        )
                    serialized_panel = json.loads((Path(budget.PANEL_DIR) / "task.json").read_text())
                    serialized_record = json.loads((Path(budget.REC_DIR) / "task.json").read_text())
                    self.assertEqual(panel["direct_fetch_enabled"], enabled)
                    self.assertEqual(serialized_panel["direct_fetch_enabled"], enabled)
                    self.assertEqual(record["direct_fetch_enabled"], enabled)
                    self.assertEqual(serialized_record["direct_fetch_enabled"], enabled)
        finally:
            budget.PANEL_DIR = original_panel_dir
            budget.REC_DIR = original_records_dir
            config.RESEARCH_ENABLE_DIRECT_FETCH = original_direct_fetch

    def test_budget_summary_serializes_effective_direct_fetch_enabled(self) -> None:
        budget = importlib.import_module("budget_fusion_full")
        original_records_dir = budget.REC_DIR
        original_direct_fetch = config.RESEARCH_ENABLE_DIRECT_FETCH
        original_api_key = config.OPENROUTER_API_KEY
        task = SimpleNamespace(id="task", domain="domain")
        record = {"task_id": task.id, "domain": task.domain, "fused": 50, "solo": {}}

        class FakeClientContext:
            async def __aenter__(self):
                return object()

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

        try:
            for enabled in (False, True):
                with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                    budget.REC_DIR = str(Path(directory) / "records")
                    config.RESEARCH_ENABLE_DIRECT_FETCH = enabled
                    with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
                         mock.patch.object(budget, "load_all", return_value=[task]), \
                         mock.patch.object(budget, "OpenRouterClient", return_value=FakeClientContext()), \
                         mock.patch.object(budget, "run_task", new=mock.AsyncMock(return_value=record)):
                        asyncio.run(budget.main())
                    serialized = json.loads(Path(budget.summary_path()).read_text())
                    self.assertEqual(serialized["direct_fetch_enabled"], enabled)
        finally:
            budget.REC_DIR = original_records_dir
            config.RESEARCH_ENABLE_DIRECT_FETCH = original_direct_fetch
            config.OPENROUTER_API_KEY = original_api_key


if __name__ == "__main__":
    unittest.main()
