"""Provider-compatible model input normalization."""
from copy import deepcopy
from typing import Any, Union


ModelInput = Union[str, list[Any]]


def normalize_messages(value: ModelInput) -> list[Any]:
    """Return an owned provider-compatible message list without inspecting entries."""
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, list):
        if not value:
            raise ValueError("messages must not be empty")
        return deepcopy(value)
    raise TypeError("input must be a prompt string or messages list")
