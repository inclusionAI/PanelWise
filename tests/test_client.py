from __future__ import annotations

import json
import unittest

import httpx

from panelwise.client import OpenAICompatibleClient
from panelwise.config import ProviderConfig
from panelwise.errors import ConfigurationError, ProviderError, ResponseError


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_parses_completion_and_usage(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["authorization"], "Bearer secret")
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "model-a")
            return httpx.Response(
                200,
                json={
                    "model": "resolved-model",
                    "choices": [{"message": {"content": "answer"}}],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 4,
                        "total_tokens": 7,
                        "cost": 0.01,
                    },
                },
            )

        provider = ProviderConfig(retries=0)
        async with OpenAICompatibleClient(
            provider, api_key="secret", transport=httpx.MockTransport(handler)
        ) as client:
            completion = await client.complete("model-a", [{"role": "user", "content": "hi"}])
        self.assertEqual(completion.content, "answer")
        self.assertEqual(completion.model, "resolved-model")
        self.assertEqual(completion.usage.total_tokens, 7)

    async def test_non_retryable_http_error(self) -> None:
        async def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "bad key"})

        async with OpenAICompatibleClient(
            ProviderConfig(retries=2),
            api_key="secret",
            transport=httpx.MockTransport(handler),
        ) as client:
            with self.assertRaises(ProviderError) as caught:
                await client.complete("model", [{"role": "user", "content": "hi"}])
        self.assertEqual(caught.exception.status_code, 401)
        self.assertFalse(caught.exception.retryable)

    async def test_rejects_malformed_success(self) -> None:
        async def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": []})

        async with OpenAICompatibleClient(
            ProviderConfig(retries=0),
            api_key="secret",
            transport=httpx.MockTransport(handler),
        ) as client:
            with self.assertRaises(ResponseError):
                await client.complete("model", [{"role": "user", "content": "hi"}])

    async def test_retries_transient_http_error(self) -> None:
        calls = 0

        async def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, json={"error": "slow down"})
            return httpx.Response(200, json={"choices": [{"message": {"content": "recovered"}}]})

        async with OpenAICompatibleClient(
            ProviderConfig(retries=1),
            api_key="secret",
            transport=httpx.MockTransport(handler),
        ) as client:
            completion = await client.complete("model", [{"role": "user", "content": "hi"}])
        self.assertEqual(completion.content, "recovered")
        self.assertEqual(calls, 2)

    def test_missing_key_is_configuration_error(self) -> None:
        with self.assertRaises(ConfigurationError):
            OpenAICompatibleClient(ProviderConfig(api_key_env="IMPOSSIBLE_PANELWISE_KEY"))


if __name__ == "__main__":
    unittest.main()
