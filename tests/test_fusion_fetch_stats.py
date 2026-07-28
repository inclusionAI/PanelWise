from __future__ import annotations

import unittest

from draco_eval.fusion import _fetch_stats
from draco_eval.research_agent import ResearchResult, TrajectoryStep


class FusionFetchStatsTests(unittest.TestCase):
    def test_fetch_stats_counts_urls_errors_and_blocks(self) -> None:
        result = ResearchResult(
            task_id="task",
            report="report",
            ok=True,
            trajectory=[
                TrajectoryStep(0, "tool_result", tool="web_search", args={"query": "x"}, result_preview="ok"),
                TrajectoryStep(
                    1,
                    "tool_result",
                    tool="web_fetch",
                    args={"url": "https://safe.example/a"},
                    result_preview="page text",
                ),
                TrajectoryStep(
                    2,
                    "tool_result",
                    tool="web_fetch",
                    args={"url": "https://datasets.huggingface.co/perplexity-ai/draco"},
                    result_preview="ERROR: blocked benchmark/rubric domain",
                ),
                TrajectoryStep(
                    3,
                    "tool_result",
                    tool="web_fetch",
                    args={"url": "https://safe.example/missing"},
                    result_preview="ERROR fetching/searching: redirect response missing Location header",
                ),
            ],
            n_searches=1,
            n_fetches=3,
            n_steps=4,
        )

        stats = _fetch_stats(result)

        self.assertEqual(stats["n_blocked_fetches"], 1)
        self.assertEqual(stats["n_fetch_errors"], 2)
        self.assertEqual(
            stats["fetched_urls"],
            [
                "https://safe.example/a",
                "https://datasets.huggingface.co/perplexity-ai/draco",
                "https://safe.example/missing",
            ],
        )
        self.assertEqual(
            stats["blocked_fetch_urls"],
            ["https://datasets.huggingface.co/perplexity-ai/draco"],
        )


if __name__ == "__main__":
    unittest.main()
