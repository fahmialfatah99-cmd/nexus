"""NEXUS core exception hierarchy.

Every error raised anywhere in the codebase derives from :class:`NexusError` so
the top level handlers can distinguish "expected, user-facing" failures from
genuine bugs (which are logged with a traceback and reported as such).
"""

from __future__ import annotations

from typing import Any, Optional


class NexusError(Exception):
    """Base class for all expected NEXUS errors."""

    #: When True the message is safe/pleasant to show to the user directly.
    user_facing = True
    #: Suggested remediation shown under the error message.
    hint: Optional[str] = None

    def __init__(self, message: str, *, hint: Optional[str] = None, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        if hint is not None:
            self.hint = hint
        self.context: dict = context

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class ConfigError(NexusError):
    """Invalid or unreadable configuration."""

    hint = "Run `nexus doctor` to inspect your configuration, or `nexus config edit`."


class AuthError(NexusError):
    """Missing/invalid credentials (HTTP 401/403)."""

    hint = "Set the API key: `nexus auth login <provider>` or export the env var."


class RateLimitError(NexusError):
    """Provider rate limit or quota exhausted (HTTP 429)."""

    def __init__(self, message: str, *, retry_after: Optional[float] = None, **context: Any) -> None:
        super().__init__(message, **context)
        self.retry_after = retry_after


class ProviderError(NexusError):
    """Upstream provider failure."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        provider: Optional[str] = None,
        retryable: bool = False,
        **context: Any,
    ) -> None:
        super().__init__(message, **context)
        self.status_code = status_code
        self.provider = provider
        self.retryable = retryable


class NetworkError(NexusError):
    """Transport-level failure (DNS, TLS, connection refused, ...)."""

    retryable = True


class TimeoutError_(NexusError):
    """Operation exceeded its deadline."""

    retryable = True


class ModelNotFoundError(NexusError):
    """Requested model is unknown to the provider."""

    hint = "Run `nexus models` to list models, or `nexus models --refresh`."


class ContextOverflowError(NexusError):
    """Conversation does not fit the model context window even after pruning."""

    hint = "Run `/compact` to summarise history, or `/clear` to start fresh."


class ToolError(NexusError):
    """A tool failed while executing (reported back to the model, not fatal)."""

    user_facing = False


class ToolNotFound(ToolError):
    """Model asked for a tool that is not registered."""


class PermissionDenied(NexusError):
    """The permission engine blocked an operation."""

    user_facing = False


class ValidationError(NexusError):
    """Data did not satisfy an expected schema/shape."""

    user_facing = False


class AbortError(NexusError):
    """User cancelled the current operation (Ctrl+C / Esc)."""

    user_facing = False


class MaxTurnsError(NexusError):
    """Agent loop hit its turn ceiling."""

    user_facing = False


class SwarmError(NexusError):
    """Swarm orchestration failure."""


__all__ = [
    "NexusError",
    "ConfigError",
    "AuthError",
    "RateLimitError",
    "ProviderError",
    "NetworkError",
    "TimeoutError_",
    "ModelNotFoundError",
    "ContextOverflowError",
    "ToolError",
    "ToolNotFound",
    "PermissionDenied",
    "ValidationError",
    "AbortError",
    "MaxTurnsError",
    "SwarmError",
]
