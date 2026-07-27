from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from draco_eval import config
from draco_eval.dataset import Task
from draco_eval.research_agent import (
    RESEARCH_SEARCH_ONLY_SYSTEM_PROMPT,
    run_research_agent,
    run_research_messages,
)


SNAPSHOTS = json.loads((Path(__file__).with_name("legacy_payload_snapshots.json")).read_text())


def _response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


class _Client:
    supports_openrouter_server_tools = True

    def __init__(self, replies: list[object]) -> None:
        self.replies = list(replies)
        self.payloads: list[dict] = []

    async def chat(self, stage: str, payload: dict) -> dict:
        self.payloads.append(copy.deepcopy(payload))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply  # type: ignore[return-value]


class ResearchMessagesTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._min_chars = config.AGENT_MIN_REPORT_CHARS
        self._max_steps = config.RESEARCH_MAX_STEPS
        self._reasoning = config.AGENT_REASONING_EFFORT
        self._direct_fetch = config.RESEARCH_ENABLE_DIRECT_FETCH
        config.AGENT_MIN_REPORT_CHARS = 1
        config.RESEARCH_MAX_STEPS = 1
        config.AGENT_REASONING_EFFORT = ""
        config.RESEARCH_ENABLE_DIRECT_FETCH = True

    def tearDown(self) -> None:
        config.AGENT_MIN_REPORT_CHARS = self._min_chars
        config.RESEARCH_MAX_STEPS = self._max_steps
        config.AGENT_REASONING_EFFORT = self._reasoning
        config.RESEARCH_ENABLE_DIRECT_FETCH = self._direct_fetch

    async def test_message_order_structured_content_and_input_immutability(self) -> None:
        source = [
            {"role": "system", "content": "caller policy", "x-provider": {"flags": [1]}},
            {"role": "user", "content": [{"type": "input_text", "text": "question"}]},
        ]
        before = copy.deepcopy(source)
        client = _Client([_response("report")])

        result = await run_research_messages(client, source, "model", [], request_id="req-1")

        self.assertTrue(result.ok)
        self.assertEqual(result.task_id, "req-1")
        self.assertEqual(client.payloads[0]["messages"], [SNAPSHOTS["research"]["messages"][0], *before])
        self.assertEqual(source, before)

    async def test_request_id_and_input_immutability_survive_finalize_failure(self) -> None:
        source = [{"role": "user", "content": [{"type": "input_text", "text": "question"}]}]
        before = copy.deepcopy(source)
        client = _Client([RuntimeError("first failure"), RuntimeError("final failure")])
        old_fails = config.RESEARCH_MAX_CONSEC_FAILS
        config.RESEARCH_MAX_CONSEC_FAILS = 1
        try:
            result = await run_research_messages(client, source, "model", [], request_id="req-fail")
        finally:
            config.RESEARCH_MAX_CONSEC_FAILS = old_fails

        self.assertFalse(result.ok)
        self.assertEqual(result.task_id, "req-fail")
        self.assertIn("final failure", result.error)
        self.assertEqual(source, before)

    async def test_task_wrapper_has_the_legacy_initial_payload(self) -> None:
        task = Task(id="task-1", domain="", problem="original task", criteria=[])
        client = _Client([_response("report")])

        result = await run_research_agent(client, task, "model", [])

        self.assertEqual(result.task_id, task.id)
        self.assertEqual(client.payloads[0], SNAPSHOTS["research"])

    async def test_default_enabled_payload_keeps_search_and_fetch_tools(self) -> None:
        client = _Client([_response("report")])

        await run_research_messages(client, "original task", "model", [])

        self.assertEqual(
            [tool["function"]["name"] for tool in client.payloads[0]["tools"]],
            ["web_search", "web_fetch"],
        )
        self.assertEqual(client.payloads[0], SNAPSHOTS["research"])

    def test_direct_fetch_defaults_to_enabled_when_environment_is_unset(self) -> None:
        root = Path(__file__).resolve().parents[1]
        env = {"PATH": os.environ["PATH"], "PYTHONPATH": str(root)}
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from draco_eval import config; assert config.RESEARCH_ENABLE_DIRECT_FETCH",
                ],
                cwd=directory,
                env=env,
                check=True,
                timeout=30,
                capture_output=True,
                text=True,
            )

    async def test_disabled_payload_is_search_only_without_fetch_prompt(self) -> None:
        config.RESEARCH_ENABLE_DIRECT_FETCH = False
        config.RESEARCH_MAX_STEPS = 2
        client = _Client([
            {"choices": [{"message": {"content": "", "tool_calls": [{
                "id": "search-1",
                "function": {"name": "web_search", "arguments": '{"query":"question"}'},
            }]}}]},
            _response("report"),
        ])

        with mock.patch(
            "draco_eval.research_agent._exec_tool",
            new=mock.AsyncMock(return_value="search results"),
        ) as exec_tool:
            result = await run_research_messages(client, "question", "model", [])

        self.assertTrue(result.ok)
        self.assertEqual(result.n_searches, 1)
        self.assertEqual(result.n_fetches, 0)
        exec_tool.assert_awaited_once()
        self.assertEqual(
            [tool["function"]["name"] for tool in client.payloads[0]["tools"]],
            ["web_search"],
        )
        self.assertEqual(client.payloads[0]["messages"][0]["content"], RESEARCH_SEARCH_ONLY_SYSTEM_PROMPT)
        self.assertNotIn("fetch", RESEARCH_SEARCH_ONLY_SYSTEM_PROMPT.lower())


if __name__ == "__main__":
    unittest.main()
