"""Public result and protocol value types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RunMode(str, Enum):
    """The two supported PanelWise execution topologies."""

    EVAL = "eval"
    TRAJECTORY = "trajectory"


@dataclass(frozen=True)
class Usage:
    """Provider-reported token usage. Missing values remain zero."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


@dataclass(frozen=True)
class Completion:
    """Provider-neutral model completion."""

    model: str
    content: str
    usage: Usage = field(default_factory=Usage)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class Candidate:
    """One panel member's independent answer or step proposal."""

    model: str
    content: str = ""
    error: str = ""
    usage: Usage = field(default_factory=Usage)

    @property
    def ok(self) -> bool:
        return bool(self.content.strip()) and not self.error


@dataclass(frozen=True)
class StepRecord:
    """One shared-trajectory decision and its observable outcome."""

    step: int
    proposals: tuple[Candidate, ...]
    decision: dict[str, Any]
    observation: str = ""
    returncode: int | None = None


@dataclass(frozen=True)
class PanelWiseResult:
    """Stable return object for both execution modes."""

    request_id: str
    mode: RunMode
    status: str
    output: str = ""
    candidates: tuple[Candidate, ...] = ()
    evaluation: dict[str, Any] = field(default_factory=dict)
    trajectory: tuple[StepRecord, ...] = ()
    artifacts: dict[str, str] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mode"] = self.mode.value
        data["ok"] = self.ok
        return data
