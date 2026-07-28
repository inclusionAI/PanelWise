from __future__ import annotations

import asyncio
import copy
import unittest
from unittest import mock

import httpx

from draco_eval import config
from draco_eval.agent import _server_tools
from draco_eval.clients import capture_responses, make_client
from draco_eval.dataset import Task
from draco_eval.openrouter import CostTracker, OpenRouterClient
from draco_eval.research_agent import _exec_tool, run_research_agent
from draco_eval.zenmux import ZenMuxClient


class FakeResponse:
    def __init__(self, status_code: int, body: dict | None = None, json_error: Exception | None = None):
        self._json_error = json_error
        request = httpx.Request("POST", "https://zenmux.test/chat/completions")
        self._response = httpx.Response(status_code, json=body or {}, request=request)

    @property
    def status_code(self) -> int:
        return self._response.status_code

    @property
    def request(self) -> httpx.Request:
        return self._response.request

    def json(self) -> dict:
        if self._json_error:
            raise self._json_error
        return self._response.json()

    def raise_for_status(self) -> None:
        self._response.raise_for_status()


class FakeAsyncClient:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = list(responses)
        self.requests: list[dict] = []

    async def post(self, path: str, json: dict) -> FakeResponse:
        self.requests.append({"path": path, "json": json})
        if not self.responses:
            raise AssertionError("unexpected extra request")
        return self.responses.pop(0)

    async def aclose(self) -> None:
        pass


class ProviderClientTests(unittest.IsolatedAsyncioTestCase):
    def test_openrouter_provider_tools_remain_search_and_fetch(self) -> None:
        original_direct_fetch = config.RESEARCH_ENABLE_DIRECT_FETCH
        config.RESEARCH_ENABLE_DIRECT_FETCH = False
        try:
            self.assertEqual(
                [tool["type"] for tool in _server_tools([])],
                ["openrouter:web_search", "openrouter:web_fetch"],
            )
        finally:
            config.RESEARCH_ENABLE_DIRECT_FETCH = original_direct_fetch

    def setUp(self) -> None:
        self._provider = config.MODEL_PROVIDER
        self._model_map = config.ZENMUX_MODEL_MAP
        self._backend = config.RESEARCH_SEARCH_BACKEND
        self._schema_fallback = config.ZENMUX_JSON_SCHEMA_FALLBACK

    async def asyncTearDown(self) -> None:
        config.MODEL_PROVIDER = self._provider
        config.ZENMUX_MODEL_MAP = self._model_map
        config.RESEARCH_SEARCH_BACKEND = self._backend
        config.ZENMUX_JSON_SCHEMA_FALLBACK = self._schema_fallback

    async def test_make_client_defaults_to_openrouter(self) -> None:
        config.MODEL_PROVIDER = "openrouter"
        client = make_client(1, CostTracker())
        try:
            self.assertIsInstance(client, OpenRouterClient)
        finally:
            await client.aclose()

    async def test_make_client_returns_zenmux(self) -> None:
        config.MODEL_PROVIDER = "zenmux"
        client = make_client(1, CostTracker())
        try:
            self.assertIsInstance(client, ZenMuxClient)
        finally:
            await client.aclose()

    async def test_zenmux_normalization_remaps_and_preserves_payload_fields(self) -> None:
        config.ZENMUX_MODEL_MAP = '{"openai/gpt-5.5":"zenmux/agent"}'
        client = ZenMuxClient(1, CostTracker())
        try:
            payload = {
                "model": "openai/gpt-5.5",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning": {"effort": "high"},
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
                "tool_choice": "auto",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "thing",
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                    },
                },
                "unknown": {"keep": True},
            }
            normalized = client._normalize_payload(payload)
            self.assertEqual(normalized["model"], "zenmux/agent")
            self.assertEqual(normalized["reasoning"], {"effort": "high"})
            self.assertNotIn("response_format", normalized)
            self.assertEqual(normalized["messages"][0]["role"], "user")
            self.assertIn("Return only a valid JSON object", normalized["messages"][0]["content"])
            self.assertIn("Do not include any fields that are not allowed by the schema", normalized["messages"][0]["content"])
            self.assertIn('"properties": {"answer": {"type": "string"}}', normalized["messages"][0]["content"])
            self.assertNotIn('"json_schema"', normalized["messages"][0]["content"])
            self.assertEqual(normalized["tools"], payload["tools"])
            self.assertEqual(normalized["tool_choice"], "auto")
            self.assertEqual(normalized["unknown"], payload["unknown"])
            self.assertIn("reasoning", payload)
            self.assertIn("response_format", payload)
            self.assertEqual(payload["messages"][0]["content"], "hi")
        finally:
            await client.aclose()

    async def test_zenmux_normalization_can_preserve_native_response_format(self) -> None:
        config.ZENMUX_JSON_SCHEMA_FALLBACK = False
        client = ZenMuxClient(1, CostTracker())
        try:
            payload = {
                "model": "model",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "json_object"},
            }
            normalized = client._normalize_payload(payload)
            self.assertEqual(normalized["response_format"], {"type": "json_object"})
            self.assertEqual(normalized["messages"], payload["messages"])
        finally:
            await client.aclose()

    async def test_openrouter_forwards_structured_multi_turn_messages_unchanged(self) -> None:
        client = OpenRouterClient(1, CostTracker())
        fake_http = FakeAsyncClient([FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})])
        client._client = fake_http
        messages = [
            {"role": "system", "content": "caller system", "provider_field": {"a": [1]}},
            {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
        ]
        before = copy.deepcopy(messages)
        try:
            await client.chat("research", {"model": "model", "messages": messages})
            self.assertEqual(fake_http.requests[0]["json"]["messages"], before)
            self.assertEqual(messages, before)
        finally:
            await client.aclose()

    async def test_zenmux_schema_fallback_preserves_conversation_and_changes_only_final_fusion_instruction(self) -> None:
        config.ZENMUX_JSON_SCHEMA_FALLBACK = True
        client = ZenMuxClient(1, CostTracker())
        fake_http = FakeAsyncClient([FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})])
        client._client = fake_http
        messages = [
            {"role": "system", "content": "PanelWise system"},
            {"role": "user", "content": [{"type": "input_text", "text": "original question"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "prior answer"}], "opaque": True},
            {"role": "user", "content": "PanelWise Fusion instruction"},
        ]
        before = copy.deepcopy(messages)
        try:
            await client.chat("fusion_judge", {
                "model": "model", "messages": messages,
                "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
            })
            normalized = fake_http.requests[0]["json"]
            self.assertEqual(normalized["messages"][:-1], before[:-1])
            self.assertEqual(normalized["messages"][-1]["content"].split("\n\nReturn only", 1)[0], before[-1]["content"])
            self.assertIn("Return only a valid JSON object", normalized["messages"][-1]["content"])
            self.assertEqual(messages, before)
        finally:
            await client.aclose()

    async def test_invalid_model_map_fails_clearly(self) -> None:
        config.ZENMUX_MODEL_MAP = "{not json"
        with self.assertRaises(SystemExit) as cm:
            ZenMuxClient(1, CostTracker())
        self.assertIn("ZENMUX_MODEL_MAP", str(cm.exception))

    async def test_response_capture_is_task_local(self) -> None:
        config.ZENMUX_MODEL_MAP = "{}"
        client = ZenMuxClient(1, CostTracker())
        try:
            async def run_one(response_id: str) -> list[dict]:
                with capture_responses() as captured:
                    await asyncio.sleep(0)
                    client._capture_response(
                        "research",
                        {"id": response_id, "usage": {"prompt_tokens": 1}},
                        "original/model",
                        "mapped/model",
                    )
                    return captured

            first, second = await asyncio.gather(run_one("gen-a"), run_one("gen-b"))
            self.assertEqual([r["response_id"] for r in first], ["gen-a"])
            self.assertEqual([r["response_id"] for r in second], ["gen-b"])
        finally:
            await client.aclose()

    async def test_response_capture_records_non_reconcilable_calls_without_id(self) -> None:
        client = ZenMuxClient(1, CostTracker())
        try:
            with capture_responses() as captured:
                client._capture_response(
                    "research",
                    {"usage": {"prompt_tokens": 1}},
                    "original/model",
                    "mapped/model",
                )
            self.assertEqual(len(captured), 1)
            self.assertEqual(captured[0]["response_id"], "")
            self.assertFalse(captured[0]["reconcilable"])
            self.assertEqual(captured[0]["usage"], {"prompt_tokens": 1})
        finally:
            await client.aclose()

    async def test_zenmux_rejects_openrouter_server_tool_backend_before_chat(self) -> None:
        class FakeZenMux:
            supports_openrouter_server_tools = False

            async def chat(self, stage: str, payload: dict) -> dict:
                raise AssertionError("chat should not be called")

        config.RESEARCH_SEARCH_BACKEND = "native"
        with self.assertRaisesRegex(RuntimeError, "RESEARCH_SEARCH_BACKEND=native"):
            await _exec_tool(FakeZenMux(), "web_search", {"query": "x"}, "model", [])

        config.RESEARCH_SEARCH_BACKEND = "exa_or"
        with self.assertRaisesRegex(RuntimeError, "RESEARCH_SEARCH_BACKEND=exa_or"):
            await _exec_tool(FakeZenMux(), "web_search", {"query": "x"}, "model", [])

    async def test_zenmux_rejects_openrouter_server_tool_backend_at_agent_entry(self) -> None:
        class FakeZenMux:
            supports_openrouter_server_tools = False

            async def chat(self, stage: str, payload: dict) -> dict:
                raise AssertionError("chat should not be called")

        config.RESEARCH_SEARCH_BACKEND = "native"
        task = Task(id="t1", domain="Finance", problem="hello", criteria=[])
        with self.assertRaisesRegex(RuntimeError, "RESEARCH_SEARCH_BACKEND=native"):
            await run_research_agent(FakeZenMux(), task, "model", [])

    async def test_zenmux_chat_retries_transient_status_and_tracks_usage(self) -> None:
        old_retries = config.HTTP_MAX_RETRIES
        config.HTTP_MAX_RETRIES = 1
        cost = CostTracker()
        client = ZenMuxClient(1, cost)
        fake_http = FakeAsyncClient([
            FakeResponse(429, {"error": {"message": "rate limited"}}),
            FakeResponse(
                200,
                {
                    "id": "gen-ok",
                    "choices": [{"message": {"content": "done"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 5, "cost": 0.25},
                },
            ),
        ])
        client._client = fake_http
        try:
            with capture_responses() as captured:
                data = await client.chat("research", {"model": "source/model", "messages": []})
            self.assertEqual(data["id"], "gen-ok")
            self.assertEqual(len(fake_http.requests), 2)
            self.assertEqual(cost.calls, 1)
            self.assertEqual(cost.prompt_tokens, 3)
            self.assertEqual(cost.completion_tokens, 5)
            self.assertEqual(cost.cost_usd, 0.25)
            self.assertEqual([r["response_id"] for r in captured], ["gen-ok"])
        finally:
            config.HTTP_MAX_RETRIES = old_retries
            await client.aclose()

    async def test_zenmux_chat_does_not_retry_non_retryable_status(self) -> None:
        old_retries = config.HTTP_MAX_RETRIES
        config.HTTP_MAX_RETRIES = 2
        client = ZenMuxClient(1, CostTracker())
        fake_http = FakeAsyncClient([FakeResponse(401, {"error": {"message": "bad key"}})])
        client._client = fake_http
        try:
            with self.assertRaises(httpx.HTTPStatusError):
                await client.chat("research", {"model": "model", "messages": []})
            self.assertEqual(len(fake_http.requests), 1)
        finally:
            config.HTTP_MAX_RETRIES = old_retries
            await client.aclose()

    async def test_zenmux_chat_does_not_retry_api_error_body(self) -> None:
        old_retries = config.HTTP_MAX_RETRIES
        config.HTTP_MAX_RETRIES = 2
        client = ZenMuxClient(1, CostTracker())
        fake_http = FakeAsyncClient([FakeResponse(200, {"error": {"message": "invalid payload"}})])
        client._client = fake_http
        try:
            with self.assertRaisesRegex(RuntimeError, "ZenMux error"):
                await client.chat("research", {"model": "model", "messages": []})
            self.assertEqual(len(fake_http.requests), 1)
        finally:
            config.HTTP_MAX_RETRIES = old_retries
            await client.aclose()


class ProviderConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._provider = config.MODEL_PROVIDER
        self._zenmux_key = config.ZENMUX_API_KEY
        self._openrouter_key = config.OPENROUTER_API_KEY
        self._backend = config.RESEARCH_SEARCH_BACKEND
        self._reasoning = config.AGENT_REASONING_EFFORT

    def tearDown(self) -> None:
        config.MODEL_PROVIDER = self._provider
        config.ZENMUX_API_KEY = self._zenmux_key
        config.OPENROUTER_API_KEY = self._openrouter_key
        config.RESEARCH_SEARCH_BACKEND = self._backend
        config.AGENT_REASONING_EFFORT = self._reasoning

    def test_zenmux_server_tool_backend_fails_validation(self) -> None:
        config.MODEL_PROVIDER = "zenmux"
        config.ZENMUX_API_KEY = "test"
        config.RESEARCH_SEARCH_BACKEND = "native"
        with self.assertRaisesRegex(SystemExit, "RESEARCH_SEARCH_BACKEND=exa"):
            config.validate_provider_config()

    def test_zenmux_exa_backend_passes_validation(self) -> None:
        config.MODEL_PROVIDER = "zenmux"
        config.ZENMUX_API_KEY = "test"
        config.RESEARCH_SEARCH_BACKEND = "exa"
        config.validate_provider_config()

    def test_zenmux_reasoning_effort_passes_validation(self) -> None:
        config.MODEL_PROVIDER = "zenmux"
        config.ZENMUX_API_KEY = "test"
        config.RESEARCH_SEARCH_BACKEND = "exa"
        config.AGENT_REASONING_EFFORT = "high"
        config.validate_provider_config()

    def test_empty_zenmux_schema_fallback_env_means_default_on(self) -> None:
        with mock.patch.dict("os.environ", {"ZENMUX_JSON_SCHEMA_FALLBACK": ""}):
            self.assertTrue(config._env_flag("ZENMUX_JSON_SCHEMA_FALLBACK", "1"))


if __name__ == "__main__":
    unittest.main()
