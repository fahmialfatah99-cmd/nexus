"""Provider registry + model catalogue.

Everything user-visible about "which backends exist" lives in this one file, so
adding a provider is a single dict entry (no code paths to touch elsewhere).

Honesty rule for the model catalogue: values we are not sure about are left as
``0`` (meaning *unknown*) instead of being invented. Unknown context windows are
never used to prune context -- the budget engine simply skips enforcement and
logs it. Run ``nexus models --refresh`` to pull the live list from the provider.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from ..core.errors import ConfigError, ModelNotFoundError, ProviderError
from .anthropic import AnthropicProvider
from .base import BaseProvider, ModelInfo
from .gemini import GeminiProvider
from .mock import MockProvider
from .openai_compat import OpenAICompatProvider


# --------------------------------------------------------------------------- #
# Provider specs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProviderSpec:
    key: str
    display_name: str
    cls: Type[BaseProvider]
    default_base_url: str = ""
    default_model: str = ""
    env_keys: Tuple[str, ...] = ()
    requires_api_key: bool = True
    docs_url: str = ""
    note: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    local: bool = False


def _compat(key: str, name: str, base_url: str, default_model: str, env: Tuple[str, ...],
            docs: str = "", note: str = "", extra: Optional[Dict[str, Any]] = None,
            local: bool = False) -> ProviderSpec:
    cls = type(
        f"{key.title().replace('_', '').replace('-', '')}Provider",
        (OpenAICompatProvider,),
        {"key": key, "display_name": name, "default_base_url": base_url, "default_model": default_model,
         "env_keys": env},
    )
    return ProviderSpec(key=key, display_name=name, cls=cls, default_base_url=base_url, default_model=default_model,
                        env_keys=env, requires_api_key=not local, docs_url=docs, note=note,
                        extra=extra or {}, local=local)


PROVIDERS: Dict[str, ProviderSpec] = {}


def _register(spec: ProviderSpec) -> ProviderSpec:
    PROVIDERS[spec.key] = spec
    return spec


_register(ProviderSpec(key="openai", display_name="OpenAI", cls=OpenAICompatProvider,
                       default_base_url="https://api.openai.com/v1", default_model="gpt-4o",
                       env_keys=("OPENAI_API_KEY",), docs_url="https://platform.openai.com/docs/api-reference"))
_register(ProviderSpec(key="anthropic", display_name="Anthropic (Claude)", cls=AnthropicProvider,
                       default_base_url="https://api.anthropic.com", default_model="claude-sonnet-4-5",
                       env_keys=("ANTHROPIC_API_KEY",), docs_url="https://docs.anthropic.com"))
_register(ProviderSpec(key="gemini", display_name="Google Gemini", cls=GeminiProvider,
                       default_base_url="https://generativelanguage.googleapis.com", default_model="gemini-2.5-pro",
                       env_keys=("GEMINI_API_KEY", "GOOGLE_API_KEY"), docs_url="https://ai.google.dev"))
_register(_compat("groq", "Groq", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile",
                  ("GROQ_API_KEY",), "https://console.groq.com/docs", "Very fast LPU inference."))
_register(_compat("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "anthropic/claude-sonnet-4.5",
                  ("OPENROUTER_API_KEY",), "https://openrouter.ai/docs", "300+ models through one key."))
_register(_compat("deepseek", "DeepSeek", "https://api.deepseek.com/v1", "deepseek-chat",
                  ("DEEPSEEK_API_KEY",), "https://api-docs.deepseek.com"))
_register(_compat("xai", "xAI (Grok)", "https://api.x.ai/v1", "grok-4-fast-reasoning",
                  ("XAI_API_KEY", "X_AI_API_KEY"), "https://docs.x.ai"))
_register(_compat("mistral", "Mistral AI", "https://api.mistral.ai/v1", "mistral-large-latest",
                  ("MISTRAL_API_KEY",), "https://docs.mistral.ai"))
_register(_compat("together", "Together AI", "https://api.together.xyz/v1", "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                  ("TOGETHER_API_KEY",), "https://docs.together.ai"))
_register(_compat("cerebras", "Cerebras", "https://api.cerebras.ai/v1", "llama-3.3-70b",
                  ("CEREBRAS_API_KEY",), "https://inference-docs.cerebras.ai"))
_register(_compat("fireworks", "Fireworks", "https://api.fireworks.ai/inference/v1",
                  "accounts/fireworks/models/llama-v3p1-70b-instruct", ("FIREWORKS_API_KEY",),
                  "https://docs.fireworks.ai"))
_register(_compat("sambanova", "SambaNova", "https://api.sambanova.ai/v1", "Meta-Llama-3.3-70B-Instruct",
                  ("SAMBANOVA_API_KEY",), "https://docs.sambanova.ai"))
_register(_compat("siliconflow", "SiliconFlow", "https://api.siliconflow.cn/v1", "Qwen/Qwen2.5-72B-Instruct",
                  ("SILICONFLOW_API_KEY",), "https://docs.siliconflow.cn"))
_register(_compat("qwen", "Alibaba Qwen (DashScope)", "https://dashscope.aliyuncs.com/compatible-mode/v1",
                  "qwen-plus", ("DASHSCOPE_API_KEY", "QWEN_API_KEY"), "https://help.aliyun.com/zh/model-studio"))
_register(_compat("moonshot", "Moonshot (Kimi)", "https://api.moonshot.cn/v1", "moonshot-v1-32k-vision-preview",
                  ("MOONSHOT_API_KEY",), "https://platform.moonshot.cn/docs"))
_register(_compat("zhipu", "Zhipu GLM", "https://open.bigmodel.cn/api/paas/v4", "glm-4-plus",
                  ("ZHIPUAI_API_KEY", "ZHIPU_API_KEY"), "https://open.bigmodel.cn/dev/api"))
_register(_compat("perplexity", "Perplexity", "https://api.perplexity.ai", "sonar-pro",
                  ("PERPLEXITY_API_KEY",), "https://docs.perplexity.ai"))
_register(_compat("huggingface", "Hugging Face Router", "https://router.huggingface.co/v1",
                  "meta-llama/Llama-3.3-70B-Instruct", ("HF_TOKEN", "HUGGINGFACE_API_KEY"),
                  "https://huggingface.co/docs/inference-providers"))
_register(_compat("github", "GitHub Models", "https://models.inference.ai.azure.com", "gpt-4o",
                  ("GITHUB_TOKEN",), "https://docs.github.com/en/github-models"))
_register(_compat("novita", "Novita", "https://api.novita.ai/v3/openai", "meta-llama/llama-3.3-70b-instruct",
                  ("NOVITA_API_KEY",), "https://novita.ai/docs"))
_register(_compat("chutes", "Chutes", "https://api.chutes.ai/v1", "deepseek-r1", ("CHUTES_API_KEY",),
                  "https://docs.chutes.ai"))
_register(_compat("ollama", "Ollama (local)", "http://localhost:11434/v1", "llama3.2",
                  (), "https://ollama.com", "Local models. No API key needed.",
                  {"stream_options": False}, local=True))
_register(_compat("lmstudio", "LM Studio (local)", "http://localhost:1234/v1", "local-model",
                  (), "https://lmstudio.ai", "Local models. No API key needed.",
                  {"stream_options": False}, local=True))
_register(_compat("vllm", "vLLM (local)", "http://localhost:8000/v1", "served-model",
                  (), "https://docs.vllm.ai", "Self-hosted OpenAI-compatible server.",
                  {"stream_options": False}, local=True))
# 9Router (github.com/decolua/9router): a local smart gateway that routes to 40+
# providers (Claude Code, Codex, Kiro, GLM, MiniMax, OpenCode Free, …) behind one
# OpenAI-compatible endpoint with auto-fallback. Port 20128 by default (PORT env).
# It does not validate the key but requires a non-empty one, so we send "local".
_register(_compat("9router", "9Router (local gateway)", "http://localhost:20128/v1",
                  "kr/claude-sonnet-4.5", ("NINEROUTER_API_KEY",),
                  "https://github.com/decolua/9router",
                  "Local multi-provider gateway with auto-fallback. Start it first, then use "
                  "`nexus models --provider 9router --refresh` to see the models it exposes.",
                  local=True))
_register(_compat("custom", "Custom endpoint", "http://localhost:8000/v1", "model",
                  ("CUSTOM_API_KEY",), "", "Any OpenAI-compatible gateway. Set base_url in config."))
_register(ProviderSpec(key="mock", display_name="Mock (offline)", cls=MockProvider, default_base_url="mock://local",
                       default_model="mock-1", requires_api_key=False, note="Deterministic offline provider for tests/demos."))

#: Friendly aliases people actually type.
ALIASES: Dict[str, str] = {
    "gpt4o": "gpt-4o", "4o": "gpt-4o", "gpt-4.1": "gpt-4.1", "o3": "o3", "o4": "o4-mini",
    "sonnet": "claude-sonnet-4-5", "opus": "claude-opus-4-1", "haiku": "claude-haiku-4-5",
    "claude": "claude-sonnet-4-5",
    "gemini": "gemini-2.5-pro", "pro": "gemini-2.5-pro", "flash": "gemini-2.5-flash",
    "deepseek": "deepseek-chat", "r1": "deepseek-reasoner", "v3": "deepseek-chat",
    "llama": "llama-3.3-70b-versatile", "grok": "grok-4-fast-reasoning", "kimi": "moonshot-v1-32k-vision-preview",
    "glm": "glm-4-plus", "sonar": "sonar-pro",
}


# --------------------------------------------------------------------------- #
# Model catalogue (best-effort public specs; 0 == unknown)
# --------------------------------------------------------------------------- #
def _m(id_: str, provider: str, ctx: int, out: int, *, tools: bool = True, images: bool = False,
       reasoning: bool = False, cin: float = 0.0, cout: float = 0.0, family: str = "") -> ModelInfo:
    return ModelInfo(id=id_, provider=provider, context_window=ctx, max_output_tokens=out, supports_tools=tools,
                     supports_images=images, supports_reasoning=reasoning, input_cost_per_mtok=cin,
                     output_cost_per_mtok=cout, family=family or id_.split("-")[0])


MODEL_CATALOG: Dict[str, ModelInfo] = {}


def _catalog(models: Sequence[ModelInfo]) -> None:
    for m in models:
        MODEL_CATALOG[m.id] = m


_catalog([
    _m("gpt-4o", "openai", 128_000, 16_384, images=True, cin=2.5, cout=10.0, family="gpt-4o"),
    _m("gpt-4o-mini", "openai", 128_000, 16_384, images=True, cin=0.15, cout=0.6, family="gpt-4o"),
    _m("gpt-4.1", "openai", 1_047_576, 32_768, images=True, cin=2.0, cout=8.0, family="gpt-4.1"),
    _m("gpt-4.1-mini", "openai", 1_047_576, 32_768, images=True, cin=0.4, cout=1.6, family="gpt-4.1"),
    _m("gpt-4.1-nano", "openai", 1_047_576, 32_768, images=True, cin=0.1, cout=0.4, family="gpt-4.1"),
    _m("o3", "openai", 200_000, 100_000, images=True, reasoning=True, cin=2.0, cout=8.0, family="o3"),
    _m("o3-mini", "openai", 200_000, 100_000, reasoning=True, cin=1.1, cout=4.4, family="o3"),
    _m("o4-mini", "openai", 200_000, 100_000, images=True, reasoning=True, cin=1.1, cout=4.4, family="o4"),
    _m("claude-opus-4-1", "anthropic", 200_000, 32_000, images=True, reasoning=True, cin=15.0, cout=75.0, family="claude-opus"),
    _m("claude-sonnet-4-5", "anthropic", 200_000, 64_000, images=True, reasoning=True, cin=3.0, cout=15.0, family="claude-sonnet"),
    _m("claude-haiku-4-5", "anthropic", 200_000, 32_000, images=True, reasoning=True, cin=1.0, cout=5.0, family="claude-haiku"),
    _m("claude-3-5-haiku-latest", "anthropic", 200_000, 8_192, cin=0.8, cout=4.0, family="claude-haiku"),
    _m("gemini-2.5-pro", "gemini", 1_048_576, 65_536, images=True, reasoning=True, cin=1.25, cout=10.0, family="gemini-2.5"),
    _m("gemini-2.5-flash", "gemini", 1_048_576, 65_536, images=True, reasoning=True, cin=0.3, cout=2.5, family="gemini-2.5"),
    _m("gemini-2.0-flash", "gemini", 1_048_576, 8_192, images=True, cin=0.1, cout=0.4, family="gemini-2.0"),
    _m("deepseek-chat", "deepseek", 64_000, 8_192, cin=0.27, cout=1.1, family="deepseek-v3"),
    _m("deepseek-reasoner", "deepseek", 64_000, 8_192, reasoning=True, cin=0.27, cout=1.1, family="deepseek-r1"),
    _m("llama-3.3-70b-versatile", "groq", 128_000, 32_768, cin=0.59, cout=0.79, family="llama-3.3"),
    _m("llama-3.1-8b-instant", "groq", 128_000, 8_192, cin=0.05, cout=0.08, family="llama-3.1"),
    _m("grok-4-fast-reasoning", "xai", 2_000_000, 30_000, images=True, reasoning=True, family="grok"),
    _m("mistral-large-latest", "mistral", 128_000, 8_192, cin=2.0, cout=6.0, family="mistral-large"),
    _m("qwen-plus", "qwen", 131_072, 16_384, cin=0.4, cout=1.2, family="qwen"),
    _m("qwen-max", "qwen", 32_768, 8_192, family="qwen"),
    _m("glm-4-plus", "zhipu", 128_000, 4_096, family="glm-4"),
    _m("sonar-pro", "perplexity", 200_000, 8_192, family="sonar"),
])


def register_models(models: Sequence[ModelInfo]) -> None:
    """Plugin/config hook: add or override catalogue entries."""
    _catalog(models)


# --------------------------------------------------------------------------- #
# Resolution helpers
# --------------------------------------------------------------------------- #
def get_spec(key: str) -> ProviderSpec:
    key = (key or "").strip().lower()
    if key in PROVIDERS:
        return PROVIDERS[key]
    # accept display names and env-var style keys
    for spec in PROVIDERS.values():
        if key and (key == spec.display_name.lower() or key in {e.lower() for e in spec.env_keys}):
            return spec
    raise ConfigError(f"Unknown provider '{key}'.", hint=f"Known providers: {', '.join(sorted(PROVIDERS))}")


def provider_keys() -> List[str]:
    return sorted(PROVIDERS)


def resolve_model_id(model: str) -> str:
    if not model:
        return model
    m = model.strip()
    return ALIASES.get(m.lower(), m)


def model_info(model_id: str, provider_key: str = "") -> ModelInfo:
    """Look up a model; never raises -- unknown models get an empty profile."""
    mid = resolve_model_id(model_id)
    if mid in MODEL_CATALOG:
        return MODEL_CATALOG[mid]
    # prefix match ("claude-sonnet-4-5-20250929" -> "claude-sonnet-4-5")
    best: Optional[ModelInfo] = None
    for known, info in MODEL_CATALOG.items():
        if mid.startswith(known) or known.startswith(mid):
            if best is None or len(known) > len(best.id):
                best = info
    if best is not None:
        clone = ModelInfo(**{**best.__dict__, "id": mid})
        return clone
    return ModelInfo(id=mid, provider=provider_key, context_window=0, max_output_tokens=0,
                     supports_images=provider_key == "gemini", family=mid.split("-")[0])


def available_providers(*, include_local: bool = False, probe_local: bool = False) -> List[ProviderSpec]:
    """Providers that are usable right now.

    ``include_local`` adds local servers (ollama/lmstudio/vllm); with
    ``probe_local`` they are only added when the port actually answers.
    """
    out = []
    for spec in PROVIDERS.values():
        if spec.local:
            if not include_local:
                continue
            if probe_local:
                host, port = _host_port(spec.default_base_url)
                if not (host and _port_open(host, port)):
                    continue
            out.append(spec)
        elif any(os.environ.get(k) for k in spec.env_keys):
            out.append(spec)
    return out


def _host_port(base_url: str) -> Tuple[str, int]:
    from urllib.parse import urlparse

    try:
        u = urlparse(base_url)
        return (u.hostname or ""), int(u.port or (443 if u.scheme == "https" else 80))
    except ValueError:
        return ("", 0)


def detect_default_provider(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    order = ["anthropic", "openai", "gemini", "groq", "openrouter", "deepseek", "qwen", "xai", "mistral", "together"]
    for key in order:
        spec = PROVIDERS.get(key)
        if spec and any(os.environ.get(k) for k in spec.env_keys):
            return key
    if os.environ.get("OLLAMA_HOST") or _port_open("127.0.0.1", 11434):
        return "ollama"
    return None


def _port_open(host: str, port: int, timeout: float = 0.15) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def create_provider(
    key: str,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    transport: Any = None,
    timeout: float = 300.0,
    max_retries: int = 4,
    headers: Optional[Dict[str, str]] = None,
    extra: Optional[Dict[str, Any]] = None,
    auth_store: Optional[Dict[str, str]] = None,
) -> BaseProvider:
    spec = get_spec(key)
    resolved_key = api_key
    if not resolved_key:
        for env in spec.env_keys:
            v = os.environ.get(env)
            if v:
                resolved_key = v
                break
    if not resolved_key and auth_store:
        resolved_key = auth_store.get(spec.key)
    merged_extra: Dict[str, Any] = dict(spec.extra or {})
    if extra:
        merged_extra.update(extra)
    try:
        return spec.cls(
            api_key=resolved_key or ("local" if not spec.requires_api_key else None),
            base_url=base_url or None,
            transport=transport,
            timeout=timeout,
            max_retries=max_retries,
            headers=headers,
            extra=merged_extra,
        )
    except ProviderError:
        raise
    except TypeError as exc:  # pragma: no cover - programming error guard
        raise ProviderError(f"Failed to instantiate provider '{key}': {exc}", provider=key) from exc


__all__ = [
    "ProviderSpec", "PROVIDERS", "ALIASES", "MODEL_CATALOG", "ModelInfo",
    "get_spec", "provider_keys", "resolve_model_id", "model_info", "register_models",
    "available_providers", "detect_default_provider", "create_provider", "ModelNotFoundError",
]
