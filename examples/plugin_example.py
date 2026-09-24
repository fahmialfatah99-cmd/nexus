"""Example NEXUS plugin.

Install by copying this file into ~/.nexus/plugins/ (personal) or
<project>/.nexus/plugins/ (shared with the team), then run `nexus plugins`
to confirm it loaded.

A plugin may expose any of:
    TOOLS      list of Tool instances
    PERSONAS   list of persona dicts
    COMMANDS   list of {"name","help","category","handler"} dicts
    register(registry, context)   a hook for anything else
"""

from __future__ import annotations

import subprocess

from nexuscli.tools.base import ConfirmationRequest, Tool, ToolContext, ToolResult


class WordCountTool(Tool):
    """A read-only tool: it never triggers an approval prompt."""

    name = "word_count"
    description = (
        "Count words, lines and characters in a workspace file. Useful for size checks "
        "before rewriting documentation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File to measure."},
        },
        "required": ["path"],
    }
    category = "examples"
    read_only = True

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        path = ctx.resolve(args["path"])
        if not path.is_file():
            return ToolResult.fail(f"No such file: {args['path']}")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult.fail(f"Cannot read {args['path']}: {exc}")
        return ToolResult.ok(
            f"{ctx.rel(path)}: {len(text.splitlines())} lines, {len(text.split())} words, "
            f"{len(text)} characters",
            lines=len(text.splitlines()), words=len(text.split()), chars=len(text),
        ).with_touched(str(path))


class TestRunnerTool(Tool):
    """A writing tool: it declares a confirmation request, so the permission
    engine decides whether to prompt based on the current approval mode."""

    name = "run_project_tests"
    description = "Run the project's test command (from .nexus/config or a detected default)."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "default": "python3 -m unittest discover -s tests"},
            "timeout": {"type": "number", "default": 300},
        },
    }
    category = "examples"

    def confirmation(self, args: dict, ctx: ToolContext) -> ConfirmationRequest:
        return ConfirmationRequest(title=f"Run tests: {args.get('command')}",
                                   detail="executes a subprocess in the workspace",
                                   risk="normal", key="run_project_tests:*")

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        command = args.get("command") or "python3 -m unittest discover -s tests"
        try:
            proc = subprocess.run(command, shell=True, cwd=str(ctx.cwd), capture_output=True,
                                  text=True, timeout=float(args.get("timeout") or 300),
                                  errors="replace")
        except subprocess.TimeoutExpired:
            return ToolResult.fail(f"Test run timed out after {args.get('timeout')}s")
        except OSError as exc:
            return ToolResult.fail(f"Could not run tests: {exc}")
        body = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        result = ToolResult(content=body.strip() or "(no output)",
                            data={"exit_code": proc.returncode})
        result.is_error = proc.returncode != 0
        return result


TOOLS = [WordCountTool(), TestRunnerTool()]

PERSONAS = [
    {
        "key": "release",
        "name": "Rocket",
        "emoji": "🚀",
        "role": "the release engineer",
        "style": "You are methodical about shipping. Nothing goes out without a changelog entry, "
                 "a version bump and a smoke test.",
        "duties": "- Prepare the release: version bump, changelog, tag.\n"
                  "- Run the full test suite and the smoke checks.\n"
                  "- Write the release notes from the actual commit list.",
        "focus": ["Is the changelog complete and accurate?", "Did the full suite pass?"],
        "temperature": 0.2,
        "model_pref": "balanced",
        "max_turns": 16,
    }
]


def register(registry, context) -> None:
    """Optional hook: called after TOOLS/PERSONAS are absorbed."""
    context.get("log").info("example plugin registered", tools=len(TOOLS))
