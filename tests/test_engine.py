from __future__ import annotations

import unittest
from typing import Any, Mapping, Sequence

from panelwise.config import config_from_dict
from panelwise.engine import PanelWise
from panelwise.executor import Observation
from panelwise.types import Completion, RunMode, Usage


def make_config(*, eval: bool = True, max_steps: int = 4):
    return config_from_dict(
        {
            "version": 1,
            "models": {
                "panel": ["model-a", "model-b", "model-c"],
                "coordinator": "coordinator",
                "evaluator": "evaluator",
            },
            "execution": {"eval": eval, "max_steps": max_steps},
        }
    )


class EvalClient:
    async def complete(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> Completion:
        del extra
        system = str(messages[0]["content"])
        if model == "evaluator":
            content = (
                '{"consensus":["shared"],"conflicts":[],"unique_insights":[],'
                '"blind_spots":[],"recommendation":"combine"}'
            )
        elif model == "coordinator" and "synthesizer" in system:
            content = "fused final answer"
        else:
            content = f"independent answer from {model}"
        return Completion(model, content, Usage(total_tokens=1))


class TrajectoryClient:
    def __init__(self) -> None:
        self.coordinator_calls = 0

    async def complete(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> Completion:
        del messages, extra
        if model != "coordinator":
            return Completion(
                model,
                '{"kind":"command","command":"printf done > result.txt","reason":"implement"}',
            )
        self.coordinator_calls += 1
        if self.coordinator_calls == 1:
            return Completion(
                model, '{"kind":"command","command":"printf done > result.txt","reason":"merge"}'
            )
        return Completion(
            model, '{"kind":"done","answer":"implemented and verified","reason":"complete"}'
        )


class PartiallyFailingEvalClient(EvalClient):
    async def complete(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> Completion:
        if model == "model-b":
            raise RuntimeError("model unavailable")
        if model == "evaluator":
            return Completion(model, "not valid JSON")
        return await super().complete(model, messages, extra=extra)


class FakeExecutor:
    def __init__(self) -> None:
        self.commands: list[str] = []

    async def observe(self) -> str:
        return "clean workspace"

    async def execute(self, command: str) -> Observation:
        self.commands.append(command)
        return Observation("created result.txt", 0)

    async def finalize(self) -> dict[str, str]:
        return {"patch": "diff --git a/result.txt b/result.txt"}


class EngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_eval_mode_fuses_independent_answers(self) -> None:
        result = await PanelWise(make_config(), EvalClient()).run("solve this", request_id="r1")
        self.assertTrue(result.ok)
        self.assertEqual(result.mode, RunMode.EVAL)
        self.assertEqual(result.output, "fused final answer")
        self.assertEqual(len(result.candidates), 3)
        self.assertEqual(result.evaluation["recommendation"], "combine")
        self.assertEqual(result.usage.total_tokens, 5)

    async def test_no_eval_mode_uses_one_shared_trajectory(self) -> None:
        executor = FakeExecutor()
        result = await PanelWise(
            make_config(eval=False), TrajectoryClient(), executor=executor
        ).run("change the workspace", request_id="r2")
        self.assertTrue(result.ok)
        self.assertEqual(result.mode, RunMode.TRAJECTORY)
        self.assertEqual(executor.commands, ["printf done > result.txt"])
        self.assertEqual(result.output, "implemented and verified")
        self.assertIn("patch", result.artifacts)
        self.assertEqual(len(result.trajectory), 2)

    async def test_eval_isolates_panel_failure_and_preserves_bad_evaluation(self) -> None:
        result = await PanelWise(make_config(), PartiallyFailingEvalClient()).run("solve this")

        self.assertTrue(result.ok)
        self.assertEqual(result.output, "fused final answer")
        self.assertEqual(result.evaluation["raw"], "not valid JSON")
        self.assertTrue(any("model-b" in error for error in result.errors))
        self.assertTrue(any("evaluator JSON" in error for error in result.errors))

    async def test_no_eval_requires_executor(self) -> None:
        with self.assertRaisesRegex(Exception, "requires an executor"):
            await PanelWise(make_config(eval=False), TrajectoryClient()).run("task")


if __name__ == "__main__":
    unittest.main()
