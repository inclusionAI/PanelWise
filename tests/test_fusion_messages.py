from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest import mock

from draco_eval import config
from draco_eval.dataset import Task
from draco_eval.fusion import (
    PanelMember,
    _analyze,
    _synthesize,
    run_fusion,
    run_fusion_messages,
)
from draco_eval.research_agent import ResearchResult, TrajectoryStep


SNAPSHOTS = json.loads((Path(__file__).with_name("legacy_payload_snapshots.json")).read_text())


def _response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


class _FusionClient:
    supports_openrouter_server_tools = True

    def __init__(self, research: dict[str, object], judge: object, synth: object, mutate_research: bool = False) -> None:
        self.research = research
        self.judge = judge
        self.synth = synth
        self.payloads: list[tuple[str, dict]] = []
        self.research_message_objects: list[object] = []
        self.mutate_research = mutate_research

    async def chat(self, stage: str, payload: dict) -> dict:
        if stage == "research":
            self.research_message_objects.append(payload["messages"][1])
            if self.mutate_research:
                payload["messages"][1]["opaque"]["items"].append(payload["model"])
        self.payloads.append((stage, copy.deepcopy(payload)))
        if stage == "research":
            reply = self.research[payload["model"]]
        elif stage == "fusion_judge":
            reply = self.judge
        else:
            reply = self.synth
        if isinstance(reply, Exception):
            raise reply
        return reply  # type: ignore[return-value]


class FusionMessagesTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._min_chars = config.AGENT_MIN_REPORT_CHARS
        self._reasoning = config.AGENT_REASONING_EFFORT
        config.AGENT_MIN_REPORT_CHARS = 1
        config.AGENT_REASONING_EFFORT = ""

    def tearDown(self) -> None:
        config.AGENT_MIN_REPORT_CHARS = self._min_chars
        config.AGENT_REASONING_EFFORT = self._reasoning

    async def test_message_path_preserves_conversation_panel_order_and_isolation(self) -> None:
        source = [
            {"role": "system", "content": "caller system", "opaque": {"items": [1]}},
            {"role": "user", "content": [{"type": "input_text", "text": "question"}]},
        ]
        before = copy.deepcopy(source)
        client = _FusionClient(
            {"first": _response("first report"), "second": _response("second report")},
            _response('{"consensus": []}'),
            _response("fused"),
        )

        result = await run_fusion_messages(client, source, ["first", "second"], "synth", "judge", [], "req-1")

        self.assertTrue(result.fused_ok)
        self.assertEqual(result.task_id, "req-1")
        self.assertEqual([member.model for member in result.panel], ["first", "second"])
        research_payloads = [payload for stage, payload in client.payloads if stage == "research"]
        self.assertEqual([p["messages"][1:] for p in research_payloads], [before, before])
        self.assertIsNot(client.research_message_objects[0], client.research_message_objects[1])
        judge = next(payload for stage, payload in client.payloads if stage == "fusion_judge")
        synth = next(payload for stage, payload in client.payloads if stage == "synthesizer")
        self.assertEqual(judge["messages"][1:-1], before)
        self.assertEqual(synth["messages"][1:-1], before)
        self.assertIn("first report", judge["messages"][-1]["content"])
        self.assertIn("second report", synth["messages"][-1]["content"])
        self.assertEqual(source, before)

    async def test_mixed_success_keeps_all_members_and_only_good_reports_reach_stages(self) -> None:
        client = _FusionClient(
            {"bad": _response(""), "good": _response("good report")},
            _response('{"consensus": []}'),
            _response("fused"),
        )
        result = await run_fusion_messages(client, "question", ["bad", "good"], "synth", "judge", [], "req")

        self.assertEqual([(member.model, member.ok) for member in result.panel], [("bad", False), ("good", True)])
        for stage, payload in client.payloads:
            if stage in {"fusion_judge", "synthesizer"}:
                self.assertIn("good report", payload["messages"][-1]["content"])
                self.assertNotIn("### Response 1 (model: bad)", payload["messages"][-1]["content"])

    async def test_analysis_failure_still_synthesizes_with_error_and_preserves_input(self) -> None:
        source = [{"role": "user", "content": [{"type": "input_text", "text": "q"}]}]
        before = copy.deepcopy(source)
        client = _FusionClient({"good": _response("report")}, RuntimeError("judge failed"), _response("fused"))

        result = await run_fusion_messages(client, source, ["good"], "synth", "judge", [], "req")

        self.assertTrue(result.fused_ok)
        self.assertEqual(result.analysis, {"_error": "judge failed"})
        synth = next(payload for stage, payload in client.payloads if stage == "synthesizer")
        self.assertIn(json.dumps({"_error": "judge failed"}, ensure_ascii=False, indent=2), synth["messages"][-1]["content"])
        self.assertEqual(source, before)

    async def test_panel_nested_content_is_independent(self) -> None:
        source = [{"role": "user", "content": "question", "opaque": {"items": [1]}}]
        before = copy.deepcopy(source)
        client = _FusionClient(
            {"first": _response("first report"), "second": _response("second report")},
            _response('{"consensus": []}'), _response("fused"), mutate_research=True,
        )

        await run_fusion_messages(client, source, ["first", "second"], "synth", "judge", [], "req")

        research_payloads = [payload for stage, payload in client.payloads if stage == "research"]
        self.assertEqual(research_payloads[0]["messages"][1]["opaque"]["items"], [1, "first"])
        self.assertEqual(research_payloads[1]["messages"][1]["opaque"]["items"], [1, "second"])
        self.assertEqual(source, before)

    async def test_synthesis_failure_returns_request_id_and_preserves_input(self) -> None:
        source = [{"role": "user", "content": [{"type": "input_text", "text": "q"}]}]
        before = copy.deepcopy(source)
        client = _FusionClient({"good": _response("report")}, _response('{"consensus": []}'), RuntimeError("synth failed"))

        result = await run_fusion_messages(client, source, ["good"], "synth", "judge", [], "req-fail")

        self.assertFalse(result.fused_ok)
        self.assertEqual(result.task_id, "req-fail")
        self.assertEqual(result.error, "synth failed")
        self.assertEqual(source, before)

    async def test_all_failed_panel_returns_request_id_and_preserves_input(self) -> None:
        source = [{"role": "user", "content": [{"type": "input_text", "text": "q"}]}]
        before = copy.deepcopy(source)
        client = _FusionClient({"bad": _response("")}, _response("unused"), _response("unused"))

        result = await run_fusion_messages(client, source, ["bad"], "synth", "judge", [], "req-none")

        self.assertFalse(result.fused_ok)
        self.assertEqual(result.task_id, "req-none")
        self.assertEqual(result.error, "no usable panel reports")
        self.assertEqual(source, before)

    async def test_panel_model_report_and_fetch_stats_stay_aligned(self) -> None:
        first = ResearchResult("req", "first", True, [
            TrajectoryStep(0, "tool_result", tool="web_fetch", args={"url": "https://one"}, result_preview="ok"),
        ], 1, 2, 3)
        second = ResearchResult("req", "", False, [
            TrajectoryStep(0, "tool_result", tool="web_fetch", args={"url": "https://two"}, result_preview="ERROR bad"),
        ], 4, 5, 6, "failed")
        client = _FusionClient({}, _response("unused"), _response("unused"))
        with mock.patch("draco_eval.fusion.run_research_messages", new=mock.AsyncMock(side_effect=[first, second])):
            result = await run_fusion_messages(client, "question", ["one", "two"], "synth", "judge", [], "req")
        self.assertEqual(
            [(member.model, member.report, member.n_searches, member.n_fetches, member.n_steps, member.fetched_urls, member.n_fetch_errors) for member in result.panel],
            [("one", "first", 1, 2, 3, ["https://one"], 0), ("two", "", 4, 5, 6, ["https://two"], 1)],
        )


class LegacyFusionPayloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_analyze_and_synthesize_payloads_are_exact(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.payloads: list[dict] = []

            async def chat(self, stage: str, payload: dict) -> dict:
                self.payloads.append(copy.deepcopy(payload))
                return _response('{"consensus": []}' if stage == "fusion_judge" else "fused")

        task = Task(id="task", domain="", problem="original", criteria=[])
        panel = [PanelMember("model", "report", True, 0, 0, 1)]
        client = Client()
        old_reasoning = config.AGENT_REASONING_EFFORT
        config.AGENT_REASONING_EFFORT = ""
        try:
            await _analyze(client, task, panel, "judge")
            await _synthesize(client, task, panel, {"consensus": []}, "synth")
        finally:
            config.AGENT_REASONING_EFFORT = old_reasoning
        self.assertEqual(client.payloads[0], SNAPSHOTS["judge"])
        self.assertEqual(client.payloads[1], SNAPSHOTS["synth"])

    async def test_legacy_run_fusion_keeps_task_path(self) -> None:
        task = Task(id="task", domain="", problem="original", criteria=[])
        result = ResearchResult("task", "report", True, [], 0, 0, 1)
        class Client:
            def __init__(self) -> None:
                self.payloads: list[dict] = []

            async def chat(self, stage: str, payload: dict) -> dict:
                self.payloads.append(copy.deepcopy(payload))
                return _response('{"consensus": []}' if stage == "fusion_judge" else "fused")

        client = Client()
        old_reasoning = config.AGENT_REASONING_EFFORT
        config.AGENT_REASONING_EFFORT = ""
        with mock.patch("draco_eval.fusion.run_research_agent", new=mock.AsyncMock(return_value=result)):
            try:
                fused = await run_fusion(client, task, ["model"], "synth", "judge", [])
            finally:
                config.AGENT_REASONING_EFFORT = old_reasoning
        self.assertTrue(fused.fused_ok)
        self.assertEqual(client.payloads[0], SNAPSHOTS["judge"])
        self.assertEqual(client.payloads[1], SNAPSHOTS["synth"])


if __name__ == "__main__":
    unittest.main()
