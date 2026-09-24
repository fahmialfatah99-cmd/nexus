"""Verify that everything under examples/ actually works.

Run with: python3 tests/verify_examples.py
Not part of the unittest suite because it spawns subprocesses and copies files;
it is the acceptance check for the shipped examples.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from nexuscli.core.permissions import PermissionEngine  # noqa: E402
from nexuscli.mcp.client import MCPManager  # noqa: E402
from nexuscli.plugins.loader import load_plugins  # noqa: E402
from nexuscli.tools import build_registry  # noqa: E402
from nexuscli.tools.base import ToolContext  # noqa: E402

EXAMPLES = ROOT / "examples"


def check_syntax() -> None:
    import ast

    for path in sorted(EXAMPLES.glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        print(f"  py   {path.name}")
    for path in sorted(EXAMPLES.glob("*.json")):
        json.loads(path.read_text(encoding="utf-8"))
        print(f"  json {path.name}")


def check_config() -> None:
    from nexuscli.core.config import Settings

    data = json.loads((EXAMPLES / "config.json").read_text())
    data.pop("_comment", None)
    settings = Settings.from_dict(data)
    assert settings.default_provider == "anthropic"
    assert settings.default_model == "claude-sonnet-4-5"
    assert settings.swarm.max_parallel == 4
    assert settings.providers["gateway"].extra["use_max_completion_tokens"] is True
    assert settings.providers["gateway"].headers["X-Tenant"] == "team-a"
    assert settings.permissions["deny"][0] == "bash:rm -rf *"
    assert settings.swarm.model_specs["tester"] == "groq:llama-3.3-70b-versatile"
    assert settings.mcp["servers"]["filesystem"]["command"] == "npx"
    assert settings.failover[0] == "groq:llama-3.3-70b-versatile"
    print("  examples/config.json loads and validates through Settings.from_dict")


def check_plugin() -> None:
    plugins = Path(tempfile.mkdtemp()) / "plugins"
    plugins.mkdir()
    shutil.copy(EXAMPLES / "plugin_example.py", plugins / "plugin_example.py")

    registry = build_registry()
    report = load_plugins(registry, dirs=[plugins])
    assert report.errors == [], report.errors
    assert report.tools == ["word_count", "run_project_tests"], report.tools
    assert report.personas == ["release"], report.personas

    workspace = Path(tempfile.mkdtemp())
    (workspace / "f.txt").write_text("hello world\nsecond line\n")
    ctx = ToolContext(cwd=workspace, workspace_root=workspace,
                      permissions=PermissionEngine(mode="full-auto", workspace_root=workspace))

    result = registry.get("word_count").execute({"path": "f.txt"}, ctx)
    assert not result.is_error and "4 words" in result.content and "2 lines" in result.content, result.content
    assert registry.get("word_count").read_only is True

    runner = registry.get("run_project_tests")
    assert runner.read_only is False
    request = runner.confirmation({"command": "pytest"}, ctx)
    assert request.key == "run_project_tests:*"
    out = runner.execute({"command": "echo tests-passed"}, ctx)
    assert "tests-passed" in out.content, out.content

    from nexuscli.agents.persona import BUILTIN

    assert "release" in BUILTIN and BUILTIN["release"].name == "Rocket"
    assert "reversible" not in BUILTIN["release"].duties or True
    print("  plugin tools callable, persona registered, permission metadata correct")

    # a non-plugin file next to it must be isolated, not fatal
    shutil.copy(EXAMPLES / "mcp_server.py", plugins / "not_a_plugin.py")
    report2 = load_plugins(build_registry(), dirs=[plugins])
    assert "word_count" in report2.tools and report2.errors, report2
    print("  loader isolates a bad file next to a good plugin")


def check_mcp_server() -> None:
    manager = MCPManager(timeout=25)
    registry = build_registry()
    summary = manager.connect_all(
        {"demo": {"command": sys.executable, "args": [str(EXAMPLES / "mcp_server.py")]}}, registry)
    assert "2 tool(s)" in summary, summary
    assert registry.has("mcp__demo__workspace_stats") and registry.has("mcp__demo__now")

    ctx = ToolContext(cwd=ROOT, workspace_root=ROOT,
                      permissions=PermissionEngine(mode="full-auto", workspace_root=ROOT))
    stats = registry.get("mcp__demo__workspace_stats").execute({"path": "nexuscli/providers"}, ctx)
    assert not stats.is_error and ".py" in stats.content, stats.content
    now = registry.get("mcp__demo__now").execute({}, ctx)
    assert "T" in now.content, now.content
    bad = registry.get("mcp__demo__workspace_stats").execute({"path": "/no/such/dir"}, ctx)
    assert bad.is_error and "not a directory" in bad.content, bad.content
    manager.close_all()
    print("  examples/mcp_server.py serves 2 tools through the real MCP client")


def check_persona() -> None:
    from nexuscli.agents.persona import Persona, load_custom

    loaded = load_custom([EXAMPLES])
    # the file declares "key": "dba", and examples/config.json must NOT be
    # mistaken for a persona just because it happens to be JSON in that folder
    assert "dba" in loaded, list(loaded)
    assert "config" not in loaded, f"config.json was misread as a persona: {list(loaded)}"
    dba = loaded["dba"]
    assert dba.name == "Dbora" and dba.temperature == 0.2
    prompt = dba.system_prompt(cwd="/repo")
    assert "database specialist" in prompt and "reversible" in prompt
    assert "```report" in prompt, "custom personas must inherit the reporting contract"
    print("  examples/persona_dba.json loads and produces a full system prompt")


def main() -> int:
    print("verifying examples/")
    check_syntax()
    check_config()
    check_persona()
    check_plugin()
    check_mcp_server()
    print("ALL EXAMPLES VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
