from __future__ import annotations

import unittest
from unittest import mock

import httpx

from draco_eval import config
from draco_eval.research_agent import _exec_tool, _url_matches_blocked_domain, _web_fetch


class _FakeAsyncClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.urls: list[str] = []

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, url: str, headers: dict[str, str]) -> httpx.Response:
        self.urls.append(url)
        if not self.responses:
            raise AssertionError("unexpected network path reached")
        return self.responses.pop(0)


class BlockedFetchSafetyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._direct_fetch = config.RESEARCH_ENABLE_DIRECT_FETCH
        config.RESEARCH_ENABLE_DIRECT_FETCH = True

    def tearDown(self) -> None:
        config.RESEARCH_ENABLE_DIRECT_FETCH = self._direct_fetch

    async def test_disabled_direct_fetch_cannot_construct_http_client(self) -> None:
        old_direct_fetch = config.RESEARCH_ENABLE_DIRECT_FETCH
        config.RESEARCH_ENABLE_DIRECT_FETCH = False
        try:
            with mock.patch("draco_eval.research_agent.httpx.AsyncClient") as async_client:
                result = await _exec_tool(
                    client=object(),
                    name="web_fetch",
                    args={"url": "https://safe.example/page"},
                    model="unused",
                    excluded_domains=[],
                )
                async_client.assert_not_called()
        finally:
            config.RESEARCH_ENABLE_DIRECT_FETCH = old_direct_fetch
        self.assertEqual(result, "ERROR: direct web_fetch is disabled by configuration")

    def test_url_matches_blocked_domain_and_subdomains(self) -> None:
        blocked = ["huggingface.co", "r2cdn.perplexity.ai"]
        self.assertTrue(_url_matches_blocked_domain("https://huggingface.co/datasets/x", blocked))
        self.assertTrue(_url_matches_blocked_domain("https://datasets.huggingface.co/x", blocked))
        self.assertTrue(_url_matches_blocked_domain("https://R2CDN.PERPLEXITY.AI/file", blocked))
        self.assertTrue(_url_matches_blocked_domain("https://r2cdn.perplexity.ai./file", blocked))

    def test_url_does_not_match_lookalike_domains(self) -> None:
        blocked = ["huggingface.co", "github.com"]
        self.assertFalse(_url_matches_blocked_domain("https://huggingface.co.evil.example/x", blocked))
        self.assertFalse(_url_matches_blocked_domain("https://notgithub.com/x", blocked))
        self.assertFalse(_url_matches_blocked_domain("not-a-url", blocked))

    async def test_web_fetch_blocks_before_network_when_enabled(self) -> None:
        old_enforced = config.ENFORCE_BLOCKED_FETCH
        old_blocked = config.BLOCKED_DOMAINS
        config.ENFORCE_BLOCKED_FETCH = True
        config.BLOCKED_DOMAINS = ["huggingface.co"]
        try:
            with mock.patch("draco_eval.research_agent.httpx.AsyncClient") as async_client:
                result = await _web_fetch("https://datasets.huggingface.co/perplexity-ai/draco")
                async_client.assert_not_called()
        finally:
            config.ENFORCE_BLOCKED_FETCH = old_enforced
            config.BLOCKED_DOMAINS = old_blocked
        self.assertEqual(result, "ERROR: blocked benchmark/rubric domain")

    async def test_web_fetch_blocks_redirect_target_when_enabled(self) -> None:
        old_enforced = config.ENFORCE_BLOCKED_FETCH
        old_blocked = config.BLOCKED_DOMAINS
        config.ENFORCE_BLOCKED_FETCH = True
        config.BLOCKED_DOMAINS = ["huggingface.co"]
        response = httpx.Response(
            302,
            headers={"location": "https://huggingface.co/datasets/perplexity-ai/draco"},
            request=httpx.Request("GET", "https://safe.example/start"),
        )
        client = _FakeAsyncClient([response])
        try:
            with mock.patch("draco_eval.research_agent.httpx.AsyncClient", return_value=client):
                result = await _web_fetch("https://safe.example/start")
        finally:
            config.ENFORCE_BLOCKED_FETCH = old_enforced
            config.BLOCKED_DOMAINS = old_blocked
        self.assertEqual(result, "ERROR: blocked benchmark/rubric domain")
        self.assertEqual(client.urls, ["https://safe.example/start"])

    async def test_web_fetch_follows_safe_redirect_when_enabled(self) -> None:
        old_enforced = config.ENFORCE_BLOCKED_FETCH
        old_blocked = config.BLOCKED_DOMAINS
        config.ENFORCE_BLOCKED_FETCH = True
        config.BLOCKED_DOMAINS = ["huggingface.co"]
        body = "<html><body><main>" + ("safe final content " * 20) + "</main></body></html>"
        redirect = httpx.Response(
            302,
            headers={"location": "https://also-safe.example/final"},
            request=httpx.Request("GET", "https://safe.example/start"),
        )
        final = httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=body.encode("utf-8"),
            request=httpx.Request("GET", "https://also-safe.example/final"),
        )
        client = _FakeAsyncClient([redirect, final])
        try:
            with mock.patch("draco_eval.research_agent.httpx.AsyncClient", return_value=client):
                result = await _web_fetch("https://safe.example/start")
        finally:
            config.ENFORCE_BLOCKED_FETCH = old_enforced
            config.BLOCKED_DOMAINS = old_blocked
        self.assertIn("safe final content", result)
        self.assertEqual(client.urls, ["https://safe.example/start", "https://also-safe.example/final"])

    async def test_web_fetch_rejects_redirect_without_location_when_enabled(self) -> None:
        old_enforced = config.ENFORCE_BLOCKED_FETCH
        old_blocked = config.BLOCKED_DOMAINS
        config.ENFORCE_BLOCKED_FETCH = True
        config.BLOCKED_DOMAINS = ["huggingface.co"]
        response = httpx.Response(
            302,
            request=httpx.Request("GET", "https://safe.example/start"),
        )
        client = _FakeAsyncClient([response])
        try:
            with mock.patch("draco_eval.research_agent.httpx.AsyncClient", return_value=client):
                with self.assertRaisesRegex(httpx.HTTPError, "missing Location"):
                    await _web_fetch("https://safe.example/start")
        finally:
            config.ENFORCE_BLOCKED_FETCH = old_enforced
            config.BLOCKED_DOMAINS = old_blocked
        self.assertEqual(client.urls, ["https://safe.example/start"])

    async def test_web_fetch_does_not_block_when_disabled(self) -> None:
        old_enforced = config.ENFORCE_BLOCKED_FETCH
        old_blocked = config.BLOCKED_DOMAINS
        config.ENFORCE_BLOCKED_FETCH = False
        config.BLOCKED_DOMAINS = ["huggingface.co"]
        try:
            with mock.patch("draco_eval.research_agent.httpx.AsyncClient") as async_client:
                async_client.side_effect = AssertionError("network path reached")
                with self.assertRaisesRegex(AssertionError, "network path reached"):
                    await _web_fetch("https://datasets.huggingface.co/perplexity-ai/draco")
                async_client.assert_called_once()
                self.assertTrue(async_client.call_args.kwargs["follow_redirects"])
        finally:
            config.ENFORCE_BLOCKED_FETCH = old_enforced
            config.BLOCKED_DOMAINS = old_blocked

    async def test_exec_tool_returns_blocked_fetch_error(self) -> None:
        old_enforced = config.ENFORCE_BLOCKED_FETCH
        old_blocked = config.BLOCKED_DOMAINS
        config.ENFORCE_BLOCKED_FETCH = True
        config.BLOCKED_DOMAINS = ["huggingface.co"]
        try:
            result = await _exec_tool(
                client=object(),
                name="web_fetch",
                args={"url": "https://datasets.huggingface.co/perplexity-ai/draco"},
                model="unused",
                excluded_domains=[],
            )
        finally:
            config.ENFORCE_BLOCKED_FETCH = old_enforced
            config.BLOCKED_DOMAINS = old_blocked
        self.assertEqual(result, "ERROR: blocked benchmark/rubric domain")


if __name__ == "__main__":
    unittest.main()
