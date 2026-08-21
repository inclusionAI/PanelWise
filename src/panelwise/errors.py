"""Stable exception hierarchy for PanelWise."""

from __future__ import annotations


class PanelWiseError(Exception):
    """Base class for documented PanelWise failures."""


class ConfigurationError(PanelWiseError):
    """The YAML file or API configuration is invalid."""


class ProviderError(PanelWiseError):
    """A model provider rejected a request or returned an HTTP failure."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class ResponseError(PanelWiseError):
    """A provider response did not match the expected completion shape."""


class ExecutionError(PanelWiseError):
    """A shared-environment action could not be executed safely."""
