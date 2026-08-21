"""Shared-environment execution adapters for trajectory mode."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .errors import ConfigurationError, ExecutionError


@dataclass(frozen=True)
class Observation:
    output: str
    returncode: int


class Executor(Protocol):
    async def observe(self) -> str: ...

    async def execute(self, command: str) -> Observation: ...

    async def finalize(self) -> dict[str, str]: ...


_UNSAFE = (
    re.compile(r"\brm\s+", re.IGNORECASE),
    re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE),
    re.compile(r"\bgit\s+clean\s+-[^\n]*f", re.IGNORECASE),
    re.compile(r"\bgit\s+checkout\s+--\b", re.IGNORECASE),
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\b(?:mkfs|shutdown|reboot|halt)\b", re.IGNORECASE),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*;\s*\}\s*;\s*:", re.IGNORECASE),
)


class LocalShellExecutor:
    """Start one bounded shell action at a time from a fixed workspace."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        timeout_seconds: float = 120.0,
        max_output_chars: int = 8_000,
        allow_unsafe_commands: bool = False,
    ) -> None:
        root = Path(workspace).expanduser().resolve()
        if not root.is_dir():
            raise ConfigurationError(f"workspace is not a directory: {root}")
        self.workspace = root
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.allow_unsafe_commands = allow_unsafe_commands

    def _validate(self, command: str) -> None:
        if not command.strip():
            raise ExecutionError("empty command rejected")
        if "\x00" in command or len(command) > 20_000:
            raise ExecutionError("invalid or excessively long command rejected")
        if not self.allow_unsafe_commands and any(pattern.search(command) for pattern in _UNSAFE):
            raise ExecutionError(
                "potentially destructive command rejected; set execution.allow_unsafe_commands "
                "only in an isolated disposable environment"
            )

    async def _run(
        self,
        command: str,
        *,
        validate: bool = True,
        truncate: bool = True,
    ) -> Observation:
        if validate:
            self._validate(command)
        env = {**os.environ, "PANELWISE_WORKSPACE": str(self.workspace)}
        process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-lc",
            command,
            cwd=str(self.workspace),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=self.timeout_seconds)
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise ExecutionError(
                f"command timed out after {self.timeout_seconds:g} seconds"
            ) from exc
        output = stdout.decode("utf-8", errors="replace")
        if truncate and len(output) > self.max_output_chars:
            output = (
                output[: self.max_output_chars]
                + f"\n...[truncated, {len(output)} characters total]"
            )
        return Observation(output=output, returncode=process.returncode or 0)

    async def observe(self) -> str:
        status = await self._run("git status --short 2>/dev/null || true", validate=False)
        detail = status.output.strip() or "clean or not a Git repository"
        return f"workspace: {self.workspace}\ninitial status:\n{detail}"

    async def execute(self, command: str) -> Observation:
        return await self._run(command)

    async def finalize(self) -> dict[str, str]:
        diff = await self._run(
            "git diff --binary 2>/dev/null || true",
            validate=False,
            truncate=False,
        )
        untracked = await self._run(
            "git ls-files --others --exclude-standard -z 2>/dev/null || true",
            validate=False,
            truncate=False,
        )
        patch_parts = [diff.output] if diff.output.strip() else []
        for path in (item for item in untracked.output.split("\x00") if item):
            new_file_diff = await self._run(
                f"git diff --no-index --binary -- /dev/null {shlex.quote(path)}",
                validate=False,
                truncate=False,
            )
            if new_file_diff.output.strip():
                patch_parts.append(new_file_diff.output)
        status = await self._run("git status --short 2>/dev/null || true", validate=False)
        artifacts = {"workspace_status": status.output}
        if patch_parts:
            artifacts["patch"] = "".join(patch_parts)
        return artifacts
