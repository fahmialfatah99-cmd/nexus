"""Model routing: resolve ``provider:model`` specs, cache clients, fail over.

Resolution order for a model string:

1. explicit ``provider:model`` -> that provider, that model
2. known catalogue entry -> its provider
3. the configured default provider
4. the only provider that has credentials configured

Failover is explicit and observable: the chain is a list of specs from config
(``failover``), and every switch is logged and reported to the UI, so you always
know which backend actually answered.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core.errors import (
    AuthError,
    ModelNotFoundError,
    NetworkError,
    NexusError,
    ProviderError,
    RateLimitError,
    TimeoutError_,
)
from ..core.logging_ import get_logger
from ..providers.base import BaseProvider, Completion, EventHandler, Message, ModelInfo, RequestOptions, ToolSpec
from ..providers.registry import (
    PROVIDERS,
    create_provider,
    get_spec,
    model_info,
    resolve_model_id,
)


#: Provider-level failures that justify trying the next backend in the chain.
#: Request-level errors (validation, context overflow) deliberately are NOT here:
#: switching provider would not fix them and would hide the real problem.
FAILOVER_ERRORS = (AuthError, RateLimitError, NetworkError, TimeoutError_, ModelNotFoundError,
                   ProviderError)


@dataclass
class RouteTarget:
    provider_key: str
    model: str
    info: ModelInfo
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"{self.provider_key}:{self.model}"


class ModelRouter:
    def __init__(
        self,
        *,
        default_provider: str = "",
        default_model: str = "",
        provider_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        failover: Optional[Sequence[str]] = None,
        auth_store: Optional[Dict[str, str]] = None,
        transport: Any = None,
        timeout: float = 300.0,
        max_retries: int = 4,
        log: Any = None,
        on_failover: Optional[Callable[[RouteTarget, RouteTarget, str], None]] = None,
    ) -> None:
        self.default_provider = default_provider
        self.default_model = default_model
        self.provider_configs = dict(provider_configs or {})
        self.failover = list(failover or [])
        self.auth_store = dict(auth_store or {})
        self.transport = transport
        self.timeout = timeout
        self.max_retries = max_retries
        self.log = log or get_logger()
        self.on_failover = on_failover
        self._cache: Dict[str, BaseProvider] = {}
        self._lock = threading.RLock()
        self._model_cache: Dict[str, List[str]] = {}
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Providers
    # ------------------------------------------------------------------ #
    def provider(self, key: str) -> BaseProvider:
        key = (key or self.default_provider or "mock").lower()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        cfg = dict(self.provider_configs.get(key) or {})
        api_key = cfg.get("api_key") or self.auth_store.get(key)
        env_key = cfg.get("api_key_env")
        if not api_key and env_key:
            api_key = os.environ.get(env_key)
        headers = dict(cfg.get("headers") or {})
        provider = create_provider(
            key,
            api_key=api_key,
            base_url=cfg.get("base_url"),
            transport=self.transport or cfg.get("transport"),
            timeout=float(cfg.get("timeout") or self.timeout),
            max_retries=int(cfg.get("max_retries") or self.max_retries),
            headers=headers,
            extra=cfg.get("extra") or {},
            auth_store=self.auth_store,
        )
        with self._lock:
            self._cache[key] = provider
        return provider

    def register_provider(self, key: str, provider: BaseProvider) -> None:
        """Pre-seed a provider instance (tests, plugins, custom transports)."""
        with self._lock:
            self._cache[key.lower()] = provider

    def drop_cached(self, key: Optional[str] = None) -> None:
        with self._lock:
            if key:
                self._cache.pop(key, None)
            else:
                self._cache.clear()

    # ------------------------------------------------------------------ #
    # Resolution
    # ------------------------------------------------------------------ #
    def resolve(self, spec: str = "") -> RouteTarget:
        """Turn a user-supplied model spec into a concrete provider + model."""
        spec = (spec or "").strip()
        if not spec:
            spec = self.default_model or ""
        provider_key = ""
        model = spec
        if ":" in spec:
            head, _, tail = spec.partition(":")
            if head.lower() in PROVIDERS and tail:
                provider_key = head.lower()
                model = tail
        model = resolve_model_id(model)
        if not model:
            if provider_key:
                model = get_spec(provider_key).default_model
            elif self.default_model:
                model = resolve_model_id(self.default_model)
            else:
                provider_key = self._detect_provider()
                model = get_spec(provider_key).default_model
        info = model_info(model, provider_key)
        if not provider_key:
            provider_key = info.provider or self.default_provider or self._detect_provider()
        if provider_key not in PROVIDERS:
            raise ModelNotFoundError(f"Unknown provider '{provider_key}' for model '{model}'.")
        return RouteTarget(provider_key=provider_key, model=model, info=info)

    def _detect_provider(self) -> str:
        from ..providers.registry import detect_default_provider

        detected = detect_default_provider(self.default_provider or None)
        if detected:
            return detected
        configured = [k for k in self.provider_configs if self.provider_configs[k].get("api_key")
                      or os.environ.get((get_spec(k).env_keys or ("",))[0])]
        if len(configured) == 1:
            return configured[0]
        return self.default_provider or "mock"

    def failover_chain(self, primary: RouteTarget) -> List[RouteTarget]:
        chain = [primary]
        seen = {primary.label}
        for spec in self.failover:
            try:
                target = self.resolve(spec)
            except NexusError:
                continue
            if target.label in seen:
                continue
            seen.add(target.label)
            chain.append(target)
        return chain

    # ------------------------------------------------------------------ #
    # Completion with failover
    # ------------------------------------------------------------------ #
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        spec: str = "",
        options: Optional[RequestOptions] = None,
        on_event: EventHandler = None,
        allow_failover: bool = True,
    ) -> Tuple[Completion, RouteTarget]:
        primary = self.resolve(spec or (options.model if options else ""))
        chain = self.failover_chain(primary) if allow_failover else [primary]
        errors: List[str] = []
        for i, target in enumerate(chain):
            opts = _clone_options(options, target)
            try:
                provider = self.provider(target.provider_key)
            except ProviderError as exc:
                errors.append(f"{target.label}: {exc}")
                continue
            try:
                result = provider.complete(messages, tools, opts, on_event=on_event)
            except FAILOVER_ERRORS as exc:
                # A dead gateway (connection refused, DNS failure, timeout) must
                # fall over exactly like a 429 does -- that is the whole point of
                # the chain, e.g. when 9Router or Ollama is not running.
                retryable = isinstance(exc, (RateLimitError, AuthError, ModelNotFoundError,
                                             NetworkError, TimeoutError_)) or getattr(exc, "retryable", False)
                errors.append(f"{target.label}: {exc}")
                self.log.warning("provider failed", target=target.label, error=str(exc), retryable=retryable)
                if i + 1 < len(chain) and retryable:
                    if self.on_failover:
                        self.on_failover(target, chain[i + 1], str(exc))
                    continue
                self.last_error = "; ".join(errors)
                raise
            except Exception as exc:  # unexpected: do not silently switch backends
                self.log.exception("unexpected provider error", exc=exc, target=target.label)
                self.last_error = str(exc)
                raise
            result.provider = target.provider_key
            result.model = result.model or target.model
            self.last_error = None
            return result, target
        self.last_error = "; ".join(errors)
        raise ProviderError(f"All providers failed: {self.last_error}")

    # ------------------------------------------------------------------ #
    def list_models(self, provider_key: str, *, refresh: bool = True, use_cache: bool = True) -> List[str]:
        if use_cache and not refresh and provider_key in self._model_cache:
            return self._model_cache[provider_key]
        try:
            models = self.provider(provider_key).list_models()
        except Exception as exc:
            self.log.warning("list_models failed", provider=provider_key, error=str(exc))
            return list(self._model_cache.get(provider_key, []))
        if models:
            self._model_cache[provider_key] = models
        return models


def _clone_options(options: Optional[RequestOptions], target: RouteTarget) -> RequestOptions:
    if options is None:
        return RequestOptions(model=target.model)
    data = {k: getattr(options, k) for k in options.__dataclass_fields__}  # type: ignore[attr-defined]
    data["model"] = target.model
    if not target.info.supports_images:
        pass  # image blocks are stripped by the caller if needed
    return RequestOptions(**data)


__all__ = ["ModelRouter", "RouteTarget", "FAILOVER_ERRORS"]
