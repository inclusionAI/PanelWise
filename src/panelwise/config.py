"""YAML-backed, validated configuration for PanelWise."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ConfigurationError
from .prompts import (
    COORDINATOR_SYSTEM,
    EVAL_PANEL_SYSTEM,
    EVALUATOR_SYSTEM,
    SYNTHESIZER_SYSTEM,
    TRAJECTORY_PANEL_SYSTEM,
)

_PROVIDER_DEFAULTS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "zenmux": ("https://zenmux.ai/api/v1", "ZENMUX_API_KEY"),
    "openai-compatible": ("", "PANELWISE_API_KEY"),
}
_RESERVED_PAYLOAD_FIELDS = {"model", "messages", "stream"}
_RESERVED_HEADERS = {"authorization", "content-type", "user-agent"}


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be a mapping")
    return dict(value)


def _request_options(value: Any, path: str) -> dict[str, Any]:
    options = _mapping(value, path)
    if not all(isinstance(key, str) for key in options):
        raise ConfigurationError(f"{path} keys must be strings")
    reserved = sorted(key for key in options if key.lower() in _RESERVED_PAYLOAD_FIELDS)
    if reserved:
        raise ConfigurationError(f"{path} cannot override: {', '.join(reserved)}")
    return options


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{path} must be a positive integer")
    return value


def _positive_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigurationError(f"{path} must be a positive number")
    return float(value)


@dataclass(frozen=True)
class ProviderConfig:
    name: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    timeout_seconds: float = 120.0
    retries: int = 2
    headers: dict[str, str] = field(default_factory=dict)
    request_options: dict[str, Any] = field(default_factory=dict)
    eval_request_options: dict[str, Any] = field(default_factory=dict)
    trajectory_request_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    panel: tuple[str, ...]
    coordinator: str
    evaluator: str


@dataclass(frozen=True)
class ExecutionConfig:
    eval: bool = True
    concurrency: int = 3
    max_steps: int = 24
    command_timeout_seconds: float = 120.0
    max_output_chars: int = 8_000
    workspace: str = "."
    allow_unsafe_commands: bool = False


@dataclass(frozen=True)
class PromptConfig:
    eval_panel: str = EVAL_PANEL_SYSTEM
    evaluator: str = EVALUATOR_SYSTEM
    synthesizer: str = SYNTHESIZER_SYSTEM
    trajectory_panel: str = TRAJECTORY_PANEL_SYSTEM
    coordinator: str = COORDINATOR_SYSTEM


@dataclass(frozen=True)
class PanelWiseConfig:
    version: int
    provider: ProviderConfig
    models: ModelConfig
    execution: ExecutionConfig
    prompts: PromptConfig = field(default_factory=PromptConfig)

    def with_eval(self, enabled: bool) -> "PanelWiseConfig":
        return replace(self, execution=replace(self.execution, eval=enabled))

    def with_workspace(self, workspace: str) -> "PanelWiseConfig":
        return replace(self, execution=replace(self.execution, workspace=workspace))


def _parse(data: Mapping[str, Any]) -> PanelWiseConfig:
    root = dict(data)
    version = root.get("version", 1)
    if version != 1:
        raise ConfigurationError("version must be 1")

    provider_data = _mapping(root.get("provider"), "provider")
    provider_name = str(provider_data.get("name", "openrouter")).strip().lower()
    if provider_name not in _PROVIDER_DEFAULTS:
        supported = ", ".join(sorted(_PROVIDER_DEFAULTS))
        raise ConfigurationError(f"provider.name must be one of: {supported}")
    default_url, default_env = _PROVIDER_DEFAULTS[provider_name]
    base_url = str(provider_data.get("base_url", default_url)).strip().rstrip("/")
    if not base_url:
        raise ConfigurationError("provider.base_url is required for openai-compatible")
    api_key_env = str(provider_data.get("api_key_env", default_env)).strip()
    if not api_key_env:
        raise ConfigurationError("provider.api_key_env must not be empty")
    retries = provider_data.get("retries", 2)
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ConfigurationError("provider.retries must be a non-negative integer")
    headers = _mapping(provider_data.get("headers"), "provider.headers")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
        raise ConfigurationError("provider.headers keys and values must be strings")
    reserved_headers = sorted(key for key in headers if key.lower() in _RESERVED_HEADERS)
    if reserved_headers:
        raise ConfigurationError(f"provider.headers cannot override: {', '.join(reserved_headers)}")
    provider = ProviderConfig(
        name=provider_name,
        base_url=base_url,
        api_key_env=api_key_env,
        timeout_seconds=_positive_float(
            provider_data.get("timeout_seconds", 120), "provider.timeout_seconds"
        ),
        retries=retries,
        headers=headers,
        request_options=_request_options(
            provider_data.get("request_options"), "provider.request_options"
        ),
        eval_request_options=_request_options(
            provider_data.get("eval_request_options"), "provider.eval_request_options"
        ),
        trajectory_request_options=_request_options(
            provider_data.get("trajectory_request_options"),
            "provider.trajectory_request_options",
        ),
    )

    models_data = _mapping(root.get("models"), "models")
    panel_value = models_data.get("panel")
    if not isinstance(panel_value, list) or len(panel_value) < 2:
        raise ConfigurationError("models.panel must contain at least two models")
    panel = tuple(str(model).strip() for model in panel_value)
    if any(not model for model in panel):
        raise ConfigurationError("models.panel entries must not be empty")
    coordinator = str(models_data.get("coordinator", "")).strip()
    if not coordinator:
        raise ConfigurationError("models.coordinator is required")
    evaluator = str(models_data.get("evaluator", coordinator)).strip()
    models = ModelConfig(panel=panel, coordinator=coordinator, evaluator=evaluator)

    execution_data = _mapping(root.get("execution"), "execution")
    eval_value = execution_data.get("eval", True)
    if not isinstance(eval_value, bool):
        raise ConfigurationError("execution.eval must be true or false")
    allow_unsafe = execution_data.get("allow_unsafe_commands", False)
    if not isinstance(allow_unsafe, bool):
        raise ConfigurationError("execution.allow_unsafe_commands must be true or false")
    workspace_value = execution_data.get("workspace", ".")
    if not isinstance(workspace_value, str) or not workspace_value.strip():
        raise ConfigurationError("execution.workspace must be a non-empty string")
    workspace = workspace_value.strip()
    execution = ExecutionConfig(
        eval=eval_value,
        concurrency=_positive_int(
            execution_data.get("concurrency", len(panel)), "execution.concurrency"
        ),
        max_steps=_positive_int(execution_data.get("max_steps", 24), "execution.max_steps"),
        command_timeout_seconds=_positive_float(
            execution_data.get("command_timeout_seconds", 120),
            "execution.command_timeout_seconds",
        ),
        max_output_chars=_positive_int(
            execution_data.get("max_output_chars", 8_000),
            "execution.max_output_chars",
        ),
        workspace=workspace,
        allow_unsafe_commands=allow_unsafe,
    )

    prompt_data = _mapping(root.get("prompts"), "prompts")
    defaults = PromptConfig()
    prompts = PromptConfig(
        eval_panel=str(prompt_data.get("eval_panel", defaults.eval_panel)),
        evaluator=str(prompt_data.get("evaluator", defaults.evaluator)),
        synthesizer=str(prompt_data.get("synthesizer", defaults.synthesizer)),
        trajectory_panel=str(prompt_data.get("trajectory_panel", defaults.trajectory_panel)),
        coordinator=str(prompt_data.get("coordinator", defaults.coordinator)),
    )
    return PanelWiseConfig(
        version=1, provider=provider, models=models, execution=execution, prompts=prompts
    )


def load_config(path: str | Path) -> PanelWiseConfig:
    """Load and validate a versioned PanelWise YAML configuration."""

    config_path = Path(path).expanduser()
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"cannot read config {config_path}: {exc}") from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigurationError("config root must be a mapping")
    return _parse(data)


def config_from_dict(data: Mapping[str, Any]) -> PanelWiseConfig:
    """Build a validated configuration without writing YAML."""

    return _parse(data)
