"""Layered configuration.

Precedence (later wins)::

    built-in defaults
    ~/.nexus/config.json            (user)
    <project>/.nexus/config.json    (project, committed)
    <project>/.nexus/config.local.json (project, git-ignored)
    NEXUS_* environment variables
    command line flags

Unknown keys produce a *warning*, never a crash: a config file from a newer
version must still load. Type errors produce a :class:`ConfigError` that names
the exact dotted path, because a silently wrong setting is worse than a loud one.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .errors import ConfigError
from .logging_ import get_logger
from .paths import config_file, home, project_config_file
from .permissions import DEFAULT_MODE, MODES

CONFIG_VERSION = 1


@dataclass
class UIConfig:
    theme: str = "dark"
    color: bool = True
    markdown: bool = True
    diff: bool = True
    spinner: bool = True
    line_numbers: bool = False
    timestamps: bool = False
    width: int = 0            # 0 = auto (terminal width)
    show_reasoning: bool = True
    show_usage: bool = True
    bell: bool = False
    compact: bool = False
    #: language for menus and dialogs: "en" or "id". Empty = autodetect from
    #: NEXUS_LANG / LANG / LC_ALL.
    language: str = ""


@dataclass
class ProviderConfig:
    api_key: str = ""
    api_key_env: str = ""
    base_url: str = ""
    models: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 0.0
    max_retries: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def clean(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in ("", 0, 0.0, {}, [])}


@dataclass
class SwarmSettings:
    mode: str = "hive"
    cast: List[str] = field(default_factory=list)
    max_parallel: int = 4
    max_rounds: int = 3
    max_turns_per_agent: int = 16
    debate_rounds: int = 2
    reviewer_gate: bool = True
    model_specs: Dict[str, str] = field(default_factory=dict)


@dataclass
class CompactionSettings:
    enabled: bool = True
    threshold: float = 0.82
    keep_recent: int = 6


@dataclass
class ProjectContextSettings:
    enabled: bool = True
    max_files: int = 400
    include_tree: bool = True
    tree_depth: int = 3
    include_git: bool = True
    read_files: List[str] = field(default_factory=lambda: ["README.md", "AGENTS.md", "NEXUS.md", "CLAUDE.md"])
    max_chars: int = 12_000


@dataclass
class Settings:
    version: int = CONFIG_VERSION
    default_provider: str = ""
    default_model: str = ""
    approval_mode: str = DEFAULT_MODE
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_turns: int = 40
    stream: bool = True
    offline: bool = False
    read_only: bool = False
    verbose: bool = False
    allow_private_network: bool = False
    extra_dirs: List[str] = field(default_factory=list)
    failover: List[str] = field(default_factory=list)
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    permissions: Dict[str, List[str]] = field(default_factory=lambda: {"allow": [], "deny": [], "ask": []})
    ui: UIConfig = field(default_factory=UIConfig)
    swarm: SwarmSettings = field(default_factory=SwarmSettings)
    compaction: CompactionSettings = field(default_factory=CompactionSettings)
    project_context: ProjectContextSettings = field(default_factory=ProjectContextSettings)
    search: Dict[str, Any] = field(default_factory=dict)
    mcp: Dict[str, Any] = field(default_factory=lambda: {"servers": {}})
    tools: Dict[str, Any] = field(default_factory=lambda: {"disabled": []})
    memory: Dict[str, Any] = field(default_factory=lambda: {"enabled": True})
    session: Dict[str, Any] = field(default_factory=lambda: {"persist": True, "keep": 50})

    # -- (de)serialisation ------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))

    @staticmethod
    def from_dict(data: Dict[str, Any], *, strict: bool = False, log: Any = None) -> "Settings":
        log = log or get_logger()
        settings = Settings()
        _apply(settings, data or {}, "", strict=strict, log=log)
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.approval_mode not in MODES:
            raise ConfigError(f"approval_mode '{self.approval_mode}' is invalid.",
                              hint=f"Valid modes: {', '.join(MODES)}")
        if self.temperature is not None and not (0.0 <= float(self.temperature) <= 2.0):
            raise ConfigError(f"temperature {self.temperature} out of range (0.0-2.0).")
        if self.max_tokens is not None and int(self.max_tokens) <= 0:
            raise ConfigError("max_tokens must be a positive integer.")
        if self.max_turns < 1:
            raise ConfigError("max_turns must be >= 1.")
        if not (0.0 < self.compaction.threshold <= 1.0):
            raise ConfigError("compaction.threshold must be in (0, 1].")
        if self.swarm.max_parallel < 1:
            raise ConfigError("swarm.max_parallel must be >= 1.")
        if self.swarm.max_rounds < 1:
            raise ConfigError("swarm.max_rounds must be >= 1.")

    # -- persistence ------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> Path:
        target = Path(path) if path else config_file()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    def provider(self, key: str) -> ProviderConfig:
        if key not in self.providers:
            self.providers[key] = ProviderConfig()
        return self.providers[key]

    def provider_configs(self) -> Dict[str, Dict[str, Any]]:
        return {k: v.clean() for k, v in self.providers.items()}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _apply(target: Any, data: Dict[str, Any], prefix: str, *, strict: bool, log: Any) -> None:
    known = {f.name: f for f in fields(target)}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if key not in known:
            if strict:
                raise ConfigError(f"Unknown configuration key '{path}'.")
            log.warning("config: ignoring unknown key", key=path)
            continue
        f = known[key]
        current = getattr(target, key)
        try:
            if isinstance(current, (UIConfig, SwarmSettings, CompactionSettings, ProjectContextSettings)):
                if not isinstance(value, dict):
                    raise ConfigError(f"'{path}' must be an object.")
                _apply(current, value, path + ".", strict=strict, log=log)
            elif key == "providers" and isinstance(value, dict):
                for pname, pdata in value.items():
                    if not isinstance(pdata, dict):
                        raise ConfigError(f"'{path}.{pname}' must be an object.")
                    pc = target.providers.get(pname) or ProviderConfig()
                    _apply(pc, pdata, f"{path}.{pname}.", strict=strict, log=log)
                    target.providers[pname] = pc
            elif isinstance(current, dict) and isinstance(value, dict):
                merged = dict(current)
                merged.update(value)
                setattr(target, key, merged)
            elif isinstance(current, list) and isinstance(value, list):
                setattr(target, key, value)
            elif current is None or f.type in ("Optional[float]", "Optional[int]") or "Optional" in str(f.type):
                setattr(target, key, _coerce_optional(value, str(f.type), path))
            elif isinstance(current, bool):
                setattr(target, key, _as_bool(value, path))
            elif isinstance(current, int):
                setattr(target, key, int(value))
            elif isinstance(current, float):
                setattr(target, key, float(value))
            elif isinstance(current, str):
                setattr(target, key, str(value))
            else:
                setattr(target, key, value)
        except ConfigError:
            raise
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid value for '{path}': {value!r} ({exc})") from exc


def _coerce_optional(value: Any, type_hint: str, path: str) -> Any:
    if value is None or value == "":
        return None
    if "float" in type_hint:
        return float(value)
    if "int" in type_hint:
        return int(value)
    return value


def _as_bool(value: Any, path: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("1", "true", "yes", "y", "on"):
            return True
        if low in ("0", "false", "no", "n", "off"):
            return False
    raise ConfigError(f"'{path}' must be a boolean, got {value!r}.")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: line {exc.lineno} column {exc.colno}: {exc.msg}",
                          hint="Validate it with `nexus doctor --json`.") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a JSON object at the top level.")
    return data


ENV_MAP = {
    "NEXUS_MODEL": ("default_model", None),
    "NEXUS_PROVIDER": ("default_provider", None),
    "NEXUS_APPROVAL_MODE": ("approval_mode", None),
    "NEXUS_MODE": ("approval_mode", None),
    "NEXUS_TEMPERATURE": ("temperature", float),
    "NEXUS_MAX_TOKENS": ("max_tokens", int),
    "NEXUS_MAX_TURNS": ("max_turns", int),
    "NEXUS_STREAM": ("stream", "bool"),
    "NEXUS_OFFLINE": ("offline", "bool"),
    "NEXUS_READ_ONLY": ("read_only", "bool"),
    "NEXUS_VERBOSE": ("verbose", "bool"),
    "NEXUS_THEME": ("ui.theme", None),
    "NEXUS_COLOR": ("ui.color", "bool"),
    "NEXUS_SWARM_MODE": ("swarm.mode", None),
    "NEXUS_SWARM_PARALLEL": ("swarm.max_parallel", int),
    "NEXUS_SWARM_ROUNDS": ("swarm.max_rounds", int),
    "NEXUS_FAILOVER": ("failover", "csv"),
    "NEXUS_BASE_URL": ("providers.custom.base_url", None),
}


def env_overrides(environ: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Translate NEXUS_* environment variables into a config-shaped dict."""
    environ = environ if environ is not None else os.environ
    out: Dict[str, Any] = {}
    for var, (path, cast) in ENV_MAP.items():
        if var not in environ:
            continue
        raw = environ[var]
        if raw == "":
            continue
        value: Any = raw
        if cast == "bool":
            value = raw.strip().lower() in ("1", "true", "yes", "on")
        elif cast == "csv":
            value = [p.strip() for p in raw.split(",") if p.strip()]
        elif cast is float:
            try:
                value = float(raw)
            except ValueError:
                continue
        elif cast is int:
            try:
                value = int(raw)
            except ValueError:
                continue
        node = out
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def load_settings(
    *,
    cwd: Optional[Path] = None,
    user_path: Optional[Path] = None,
    project_path: Optional[Path] = None,
    environ: Optional[Dict[str, str]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    strict: bool = False,
    log: Any = None,
) -> Settings:
    cwd = Path(cwd) if cwd else Path.cwd()
    log = log or get_logger()
    merged: Dict[str, Any] = {}
    sources: List[str] = []
    user_cfg = Path(user_path) if user_path else config_file()
    if user_cfg.is_file():
        merged = deep_merge(merged, load_json(user_cfg))
        sources.append(str(user_cfg))
    proj_cfg = Path(project_path) if project_path else project_config_file(cwd)
    if proj_cfg.is_file():
        merged = deep_merge(merged, load_json(proj_cfg))
        sources.append(str(proj_cfg))
    local_cfg = proj_cfg.with_name("config.local.json")
    if local_cfg.is_file():
        merged = deep_merge(merged, load_json(local_cfg))
        sources.append(str(local_cfg))
    env_cfg = env_overrides(environ)
    if env_cfg:
        merged = deep_merge(merged, env_cfg)
        sources.append("env:NEXUS_*")
    if overrides:
        merged = deep_merge(merged, overrides)
        sources.append("cli")
    settings = Settings.from_dict(merged, strict=strict, log=log)
    settings.__dict__["_sources"] = sources  # informational, not serialised
    if settings.read_only:
        settings.approval_mode = "read-only"
    return settings


def config_sources(settings: Settings) -> List[str]:
    return list(getattr(settings, "_sources", []))


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
def load_auth() -> Dict[str, str]:
    from .paths import auth_file

    path = auth_file()
    data = load_json(path)
    keys = data.get("keys") if isinstance(data.get("keys"), dict) else {}
    return {str(k): str(v) for k, v in keys.items()}


def _write_private_json(path: Path, data: dict) -> None:
    """Write *data* as JSON to *path* with owner-only (0600) permissions.

    The file is created via ``os.open`` with mode 0o600 so the secret payload
    is never briefly readable by other users; if the file already exists its
    mode is tightened *before* the contents are replaced.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(data, indent=2) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)
    try:
        os.chmod(path, 0o600)  # covers pre-existing files created with wider modes
    except OSError:
        pass


def save_auth_key(provider: str, api_key: str) -> Path:
    from .paths import auth_file

    path = auth_file()
    data = load_json(path)
    keys = data.get("keys") if isinstance(data.get("keys"), dict) else {}
    keys[provider] = api_key
    data["keys"] = keys
    _write_private_json(path, data)
    return path


def remove_auth_key(provider: str) -> bool:
    from .paths import auth_file

    path = auth_file()
    data = load_json(path)
    keys = data.get("keys") if isinstance(data.get("keys"), dict) else {}
    if provider not in keys:
        return False
    del keys[provider]
    data["keys"] = keys
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return True


__all__ = [
    "Settings", "UIConfig", "ProviderConfig", "SwarmSettings", "CompactionSettings",
    "ProjectContextSettings", "load_settings", "load_json", "deep_merge", "env_overrides",
    "config_sources", "load_auth", "save_auth_key", "remove_auth_key", "CONFIG_VERSION",
]
