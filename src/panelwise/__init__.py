"""PanelWise public Python API."""

from ._version import __version__
from .client import ChatClient, OpenAICompatibleClient
from .config import (
    ExecutionConfig,
    ModelConfig,
    PanelWiseConfig,
    PromptConfig,
    ProviderConfig,
    config_from_dict,
    load_config,
)
from .engine import PanelWise
from .errors import (
    ConfigurationError,
    ExecutionError,
    PanelWiseError,
    ProviderError,
    ResponseError,
)
from .executor import Executor, LocalShellExecutor, Observation
from .types import Candidate, Completion, PanelWiseResult, RunMode, StepRecord, Usage

__all__ = [
    "Candidate",
    "ChatClient",
    "Completion",
    "ConfigurationError",
    "ExecutionError",
    "ExecutionConfig",
    "Executor",
    "LocalShellExecutor",
    "ModelConfig",
    "Observation",
    "OpenAICompatibleClient",
    "PanelWise",
    "PanelWiseConfig",
    "PanelWiseError",
    "PanelWiseResult",
    "PromptConfig",
    "ProviderError",
    "ProviderConfig",
    "ResponseError",
    "RunMode",
    "StepRecord",
    "Usage",
    "__version__",
    "config_from_dict",
    "load_config",
]
