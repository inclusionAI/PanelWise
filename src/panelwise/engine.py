"""The two PanelWise aggregation topologies behind one public API."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from .client import ChatClient, client_from_config
from .config import PanelWiseConfig, load_config
from .errors import ConfigurationError, ExecutionError, PanelWiseError, ResponseError
from .executor import Executor, LocalShellExecutor
from .types import Candidate, PanelWiseResult, RunMode, StepRecord, Usage

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_BASH_FENCE = re.compile(r"```(?:bash|sh)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fence = _JSON_FENCE.search(candidate)
    if fence:
        candidate = fence.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ResponseError(f"expected a JSON object: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ResponseError("expected a JSON object")
    return value


def _action(text: str) -> dict[str, Any]:
    try:
        value = _json_object(text)
    except ResponseError:
        fence = _BASH_FENCE.search(text)
        if fence:
            return {"kind": "command", "command": fence.group(1).strip(), "reason": ""}
        raise
    kind = str(value.get("kind", "")).strip().lower()
    if kind == "command" and isinstance(value.get("command"), str):
        return {
            "kind": "command",
            "command": value["command"].strip(),
            "reason": str(value.get("reason", "")),
        }
    if kind == "done":
        return {
            "kind": "done",
            "answer": str(value.get("answer", "")).strip(),
            "reason": str(value.get("reason", "")),
        }
    raise ResponseError("action must use kind=command with command, or kind=done with answer")


def _candidate_block(candidates: Sequence[Candidate]) -> str:
    return "\n\n".join(
        f"## Candidate {index}: {candidate.model}\n{candidate.content}"
        for index, candidate in enumerate(candidates, 1)
        if candidate.ok
    )


class PanelWise:
    """Run evaluated-answer fusion or shared-trajectory fusion."""

    def __init__(
        self,
        config: PanelWiseConfig,
        client: ChatClient,
        *,
        executor: Executor | None = None,
        owns_client: bool = False,
    ) -> None:
        self.config = config
        self.client = client
        self.executor = executor
        self._owns_client = owns_client

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        api_key: str | None = None,
        eval: bool | None = None,
        workspace: str | None = None,
        executor: Executor | None = None,
    ) -> "PanelWise":
        """Create a ready-to-run engine from YAML and provider environment variables."""

        config = load_config(path)
        if eval is not None:
            config = config.with_eval(eval)
        if workspace is not None:
            config = config.with_workspace(workspace)
        if executor is None and not config.execution.eval:
            executor = LocalShellExecutor(
                config.execution.workspace,
                timeout_seconds=config.execution.command_timeout_seconds,
                max_output_chars=config.execution.max_output_chars,
                allow_unsafe_commands=config.execution.allow_unsafe_commands,
            )
        return cls(
            config,
            client_from_config(config.provider, api_key=api_key),
            executor=executor,
            owns_client=True,
        )

    async def __aenter__(self) -> "PanelWise":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close a provider client created by :meth:`from_yaml`."""

        if not self._owns_client:
            return
        close = getattr(self.client, "aclose", None)
        if close is not None:
            await close()

    async def run(
        self,
        task: str,
        *,
        eval: bool | None = None,
        request_id: str | None = None,
    ) -> PanelWiseResult:
        """Run an arbitrary task with the selected topology.

        ``eval=True`` runs independent full attempts, evaluation, and synthesis.
        ``eval=False`` runs iterative proposals against a shared executor.
        """

        if not isinstance(task, str) or not task.strip():
            raise ConfigurationError("task must be a non-empty string")
        enabled = self.config.execution.eval if eval is None else eval
        rid = request_id or uuid.uuid4().hex
        if enabled:
            return await self._run_eval(task.strip(), rid)
        if self.executor is None:
            raise ConfigurationError("--no-eval requires an executor or workspace")
        return await self._run_trajectory(task.strip(), rid)

    async def _panel_call(
        self,
        model: str,
        system: str,
        user: str,
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> Candidate:
        try:
            completion = await self.client.complete(
                model,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                extra=extra,
            )
            return Candidate(model=model, content=completion.content, usage=completion.usage)
        except PanelWiseError as exc:
            return Candidate(model=model, error=str(exc))
        except Exception as exc:  # custom clients may raise their own errors
            return Candidate(model=model, error=f"{type(exc).__name__}: {exc}")

    async def _gather_panel(
        self,
        system: str,
        user: str,
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> tuple[Candidate, ...]:
        semaphore = asyncio.Semaphore(self.config.execution.concurrency)

        async def bounded(model: str) -> Candidate:
            async with semaphore:
                return await self._panel_call(model, system, user, extra=extra)

        return tuple(await asyncio.gather(*(bounded(model) for model in self.config.models.panel)))

    async def _run_eval(self, task: str, request_id: str) -> PanelWiseResult:
        eval_options = self.config.provider.eval_request_options
        candidates = await self._gather_panel(
            self.config.prompts.eval_panel, task, extra=eval_options
        )
        successful = tuple(candidate for candidate in candidates if candidate.ok)
        usage = sum((candidate.usage for candidate in candidates), Usage())
        errors = [
            f"{candidate.model}: {candidate.error}" for candidate in candidates if candidate.error
        ]
        if not successful:
            return PanelWiseResult(
                request_id=request_id,
                mode=RunMode.EVAL,
                status="failed",
                candidates=candidates,
                usage=usage,
                errors=tuple(errors or ["all panel members failed"]),
            )

        block = _candidate_block(successful)
        evaluation: dict[str, Any] = {}
        try:
            judged = await self.client.complete(
                self.config.models.evaluator,
                [
                    {"role": "system", "content": self.config.prompts.evaluator},
                    {
                        "role": "user",
                        "content": f"Original task:\n{task}\n\nIndependent attempts:\n{block}",
                    },
                ],
                extra=eval_options,
            )
            usage = usage + judged.usage
            try:
                evaluation = _json_object(judged.content)
            except ResponseError as exc:
                evaluation = {"raw": judged.content, "parse_error": str(exc)}
                errors.append(f"evaluator JSON: {exc}")
        except Exception as exc:
            errors.append(f"evaluator: {type(exc).__name__}: {exc}")

        try:
            synthesized = await self.client.complete(
                self.config.models.coordinator,
                [
                    {"role": "system", "content": self.config.prompts.synthesizer},
                    {
                        "role": "user",
                        "content": (
                            f"Original task:\n{task}\n\nIndependent attempts:\n{block}\n\n"
                            f"Evaluator analysis:\n{json.dumps(evaluation, ensure_ascii=False)}"
                        ),
                    },
                ],
                extra=eval_options,
            )
            usage = usage + synthesized.usage
            output = synthesized.content
            status = "completed"
        except Exception as exc:
            errors.append(f"synthesizer: {type(exc).__name__}: {exc}")
            output = successful[0].content
            status = "partial"

        return PanelWiseResult(
            request_id=request_id,
            mode=RunMode.EVAL,
            status=status,
            output=output,
            candidates=candidates,
            evaluation=evaluation,
            usage=usage,
            errors=tuple(errors),
        )

    async def _run_trajectory(self, task: str, request_id: str) -> PanelWiseResult:
        assert self.executor is not None
        observation = await self.executor.observe()
        records: list[StepRecord] = []
        errors: list[str] = []
        usage = Usage()
        answer = ""
        done = False

        for step_number in range(1, self.config.execution.max_steps + 1):
            recent = records[-8:]
            history = "\n\n".join(
                f"Step {record.step}: {json.dumps(record.decision, ensure_ascii=False)}\n"
                f"Result (exit {record.returncode}):\n{record.observation}"
                for record in recent
            )
            user = (
                f"Task:\n{task}\n\nCurrent observation:\n{observation}\n\n"
                f"Recent shared trajectory:\n{history or '(none)'}"
            )
            trajectory_options = self.config.provider.trajectory_request_options
            proposals = await self._gather_panel(
                self.config.prompts.trajectory_panel,
                user,
                extra=trajectory_options,
            )
            usage = usage + sum((proposal.usage for proposal in proposals), Usage())
            valid = tuple(proposal for proposal in proposals if proposal.ok)
            errors.extend(
                f"step {step_number} {proposal.model}: {proposal.error}"
                for proposal in proposals
                if proposal.error
            )
            if not valid:
                errors.append(f"step {step_number}: all panel proposals failed")
                break

            proposal_text = "\n\n".join(
                f"## {proposal.model}\n{proposal.content}" for proposal in valid
            )
            try:
                coordinated = await self.client.complete(
                    self.config.models.coordinator,
                    [
                        {"role": "system", "content": self.config.prompts.coordinator},
                        {
                            "role": "user",
                            "content": f"{user}\n\nPanel proposals:\n{proposal_text}",
                        },
                    ],
                    extra=trajectory_options,
                )
                usage = usage + coordinated.usage
                decision = _action(coordinated.content)
            except Exception as exc:
                errors.append(f"step {step_number} coordinator: {type(exc).__name__}: {exc}")
                break

            if decision["kind"] == "done":
                answer = decision.get("answer", "")
                records.append(StepRecord(step_number, proposals, decision, observation, None))
                done = True
                break

            command = decision.get("command", "")
            try:
                result = await self.executor.execute(command)
                observation = result.output or "(command produced no output)"
                returncode: int | None = result.returncode
            except ExecutionError as exc:
                observation = f"[execution rejected: {exc}]"
                returncode = None
                errors.append(f"step {step_number}: {exc}")
            records.append(StepRecord(step_number, proposals, decision, observation, returncode))

        artifacts = await self.executor.finalize()
        if not answer and artifacts.get("patch"):
            answer = "Shared execution produced the patch recorded in result.artifacts['patch']."
        status = (
            "completed" if done else ("partial" if answer or artifacts.get("patch") else "failed")
        )
        return PanelWiseResult(
            request_id=request_id,
            mode=RunMode.TRAJECTORY,
            status=status,
            output=answer,
            trajectory=tuple(records),
            artifacts=artifacts,
            usage=usage,
            errors=tuple(errors),
        )
