"""Claude Agent SDK model definition."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import logging
import shutil
import time

import asyncio
import claude_agent_sdk
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions, SdkPluginConfig,
    ToolPermissionContext, PermissionResultAllow, PermissionResultDeny
)
from claude_agent_sdk.types import McpStdioServerConfig, SystemPromptPreset

from typing import Any, Iterable, Optional

from dataclasses import dataclass, field, asdict
from core.problem import Problem
from core.models.base_agentic import BaseAgentic, AgenticConfig, RetryGenerationException
from core import var


def get_lib_version(lib_name: str) -> str:
    """Get the version of a library using uvx."""
    return subprocess.run(
        ['uvx', '--from', lib_name, 'python', '-c',
         f"from importlib.metadata import version; print(version('{lib_name}'))"],
        capture_output=True, text=True
    ).stdout.strip()


CLAUDE_BIN_PATH = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
CLAUDE_VERSION = subprocess.run([str(CLAUDE_BIN_PATH), "--version"],
                                capture_output=True, text=True).stdout.split(" ")[0]

CLAUDE_STATUS_PROMPT = {
    'base': "Reply with a single word among the flags [OK|FAIL|ISSUE], reporting the outcome of the task:\n"
            "* OK — task completed, no sorry remains, code compiles\n"
            "* FAIL — unable to complete, sorry remains or code does not compile\n"
            "* ISSUE — task is impossible due to bugs/incorrect assumptions, and rigorous arguments in support have been provided",
    'mismatch-spec': "Reply with a single word among the flags [OK|FAIL|MISMATCH|ISSUE], reporting the outcome of the task:\n"
                     "* OK — task completed, no sorry remains, code compiles\n"
                     "* FAIL — unable to complete, sorry remains or code does not compile\n"
                     "* MISMATCH — equivalence is unprovable because the specifications differ, yet both are valid interpretations of the NL description and test cases; rigorous arguments in support have been provided\n"
                     "* ISSUE — equivalence is unprovable because at least one specification is erroneous (contradicts the NL description or test cases); rigorous arguments in support have been provided",
    'mismatch-spec-mult': "Reply with a line formatted as \"<index>:<flag>\" for each file 'problem_<index>.lean', with the flag among [OK|FAIL|MISMATCH|ISSUE], reporting the outcome of the task on that file:\n"
                          "* OK — task completed, no sorry remains, code compiles\n"
                          "* FAIL — unable to complete, sorry remains or code does not compile\n"
                          "* MISMATCH — equivalence is unprovable because the specifications differ, yet both are valid interpretations of the NL description and test cases; rigorous arguments in support have been provided\n"
                          "* ISSUE — equivalence is unprovable because at least one specification is erroneous (contradicts the NL description or test cases); rigorous arguments in support have been provided",
    'mismatch-impl': "Reply with a single word among the flags [OK|FAIL|MISMATCH|ISSUE], reporting the outcome of the task:\n"
                     "* OK — task completed, no sorry remains, code compiles\n"
                     "* FAIL — unable to complete, sorry remains or code does not compile\n"
                     "* MISMATCH — equivalence is unprovable because the specification/implementation differ, yet both are valid interpretations of the NL description and test cases; rigorous arguments in support have been provided\n"
                     "* ISSUE — equivalence is unprovable because the specification, the implementation or both are erroneous (contradicting the NL description or test cases); rigorous arguments in support have been provided",
}

CLAUDE_USAGE_LIMIT_REPLY = "You've hit your limit"
CLAUDE_USAGE_LIMIT_RETRY_DELAY = 10 * 60  # 10 minutes
CLAUDE_USAGE_LIMIT_RETRY_MAX_DELAY = 60 * 60  # 1 hour


def _detect_usage_limit(message: str):
    if message is None:
        return
    if CLAUDE_USAGE_LIMIT_REPLY in message:
        raise RetryGenerationException(message=f"Claude usage limit hit: '{message}'",
                                       wait_seconds=CLAUDE_USAGE_LIMIT_RETRY_DELAY)


def _raise_for_rate_limit_event(message: dict[str, Any]) -> None:
    if message.get('type', None) != "RateLimitEvent":
        return

    info = message.get('rate_limit_info', dict())
    if info.get('status', None) != "rejected" or info.get('overage_status', None) == "allowed":
        return

    resets_at = info.get('resets_at', None)
    if resets_at is None:
        wait_seconds = CLAUDE_USAGE_LIMIT_RETRY_DELAY
    else:
        wait_seconds = max(CLAUDE_USAGE_LIMIT_RETRY_DELAY, int(resets_at - time.time()) + 1)

    raise RetryGenerationException(message=f"Claude usage limit hit: '{message}'",
                                   wait_seconds=min(wait_seconds, CLAUDE_USAGE_LIMIT_RETRY_MAX_DELAY))


@dataclass
class ClaudeSkillConfig:
    """Configuration for a single Claude skill."""
    root: str
    subdir: Path | str
    branch: str
    commit: str


MCP_SERVER_CLASSES = {"stdio": McpStdioServerConfig, }


@dataclass
class ClaudeMcpServerConfig:  # TODO: move directly to McpStdioServerConfig
    """Configuration for a single MCP server."""
    type: str
    command: str | None = None
    args: list[str] | None = None


@dataclass
class ContextConfig:
    """Configuration for the context provided to the Claude agent."""
    src: Optional[Path] = None
    dst: Optional[Path] = Path("task.md")
    prompt_type: Optional[str] = "preset"
    prompt_preset: Optional[str] = "claude_code"


@dataclass
class ClaudeConfig(AgenticConfig):
    """Configuration for the Claude agent."""
    skills: dict[str, ClaudeSkillConfig] = field(default_factory=lambda: {
        'lean4-skills': ClaudeSkillConfig(root="freer-skills", subdir="plugins/lean4",
                                          branch="main", commit="a1cef7dc21d9b68b79311d57cd78ecf6b7ca22bf")
    })
    mcp_server: dict[str, ClaudeMcpServerConfig] = field(default_factory=lambda: {
        "lean-lsp": ClaudeMcpServerConfig(type="stdio", command="uvx", args=["lean-lsp-mcp"])
    })
    context: Optional[ContextConfig] = None
    prompt: str = ""
    status_prompt: Optional[str] = 'base'
    model: Optional[str] = "claude-opus-4-6[1m]"
    dependencies: dict[str, str] = field(default_factory=lambda: {
        'lean-lsp-mcp': get_lib_version('lean-lsp-mcp'), 'leanclient': get_lib_version('leanclient')
    })


TOOLS_PATH_ATTRIBUTES = {
    "Read": ("read", "file_path"), "Edit": ("write", "file_path"), "Write": ("write", "file_path"),
    "Glob": ("read", "path"), "Grep": ("read", "path"),
    "NotebookEdit": ("write", "notebook_path"),  # Not really necessary...
}


def parse_message(message) -> dict[str, Any]:
    return {'type': type(message).__name__, **asdict(message)}


class Claude(BaseAgentic):
    """Wrapper for the Claude Agent SDK theorem prover."""
    PROVER_NAME = "claude"
    CONFIG_CLASS = ClaudeConfig

    async def claude(self, problem: Problem, prompt: str, paths: dict[str, Path]) -> tuple[
        list[dict[str, Any]], Optional[str]]:
        """Run the Claude agent with sandboxed file-system permissions and stream the response."""
        allowed_paths = self._get_allowed_paths(paths)
        async def permission_handler(tool_name: str, input_data: dict, context: ToolPermissionContext):
            mode, attr = TOOLS_PATH_ATTRIBUTES.get(tool_name, (None, None))
            if mode is None:
                return PermissionResultAllow(updated_input=input_data)
            path = input_data.get(attr, None)
            allowed = allowed_paths.get(mode, ())
            if not any(path.startswith(start) for start in allowed):
                logging.warning(
                    f"Claude tool '{tool_name}' tried to {mode} restricted path {path}")
                return PermissionResultDeny(
                    message=f"{mode.capitalize()} access to {path} denied: outside of allowed directories {allowed}",
                    interrupt=False)
            else:
                logging.debug(
                    f"Allowed tool '{tool_name}' to {mode} path {path}")
                return PermissionResultAllow(updated_input=input_data)

        if self.config.context is not None:
            src = var.CONTEXTS_DIRECTORY / self.config.context.src
            system_prompt_opt = SystemPromptPreset(type=self.config.context.prompt_type,
                                                   preset=self.config.context.prompt_preset,
                                                   append=src.read_text())
            if self.config.context.dst is not None:
                shutil.copy(src, paths['env'] / self.config.context.dst)
        else:
            system_prompt_opt = None
        options = ClaudeAgentOptions(
            cwd=paths['env'], permission_mode='default', can_use_tool=permission_handler,
            model=self.config.model,
            system_prompt=system_prompt_opt,
            plugins=[
                SdkPluginConfig(type="local", path=str(Path(paths['skills']) / config.root / config.subdir))
                for key, config in self.config.skills.items()
            ],
            mcp_servers={
                key: MCP_SERVER_CLASSES[config.type](**asdict(config),
                                                     # env={"LEAN_PROJECT_PATH": paths['env'], "LEAN_LOG_LEVEL": "DEBUG"}
                                                     ) for key, config in self.config.mcp_server.items()
            }
        )
        trace, status = list(), None
        problem.reset_timer()
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                msg = parse_message(message)
                _raise_for_rate_limit_event(msg)
                trace.append(msg)
                logging.debug(f"New message: {trace[-1]}")
                if status != "TIMEOUT" and problem.get_elapsed() > self.config.generation_timeout:
                    await client.interrupt()
                    logging.warning(
                        f"Claude generation timed out after {self.config.generation_timeout} seconds.")
                    status = "TIMEOUT"
            if self.config.status_prompt is not None:
                await client.query(CLAUDE_STATUS_PROMPT[self.config.status_prompt])
                async for message in client.receive_response():
                    msg = parse_message(message)
                    _raise_for_rate_limit_event(msg)
                    if msg.get('type', None) == "ResultMessage" and status is None:
                        status = msg['result']
                    else:
                        logging.debug(f"Received non-result message while waiting for status: {msg}")
                        trace.append(msg)
        return trace, status

    def _agent(self, problem: Problem, paths: dict[str, Path | list[Path]] | None = None, **kwargs: Any) -> None:
        """Build the prompt from the sorry location and invoke the Claude agent."""
        # line = problem.text.split("sorry")[0].count('\n') + 1  # FIXME: Returns only first sorry found
        prompt = self.config.prompt.format(filepath=" ".join(s.name for s in paths['problems']))
        trace, status = asyncio.run(self.claude(problem=problem, prompt=prompt, paths=paths))
        problem.metadata.update({'prompt': prompt, 'claude': trace, 'status': status})
        _detect_usage_limit(status)
        try:
            result = next(t['result'] for t in trace if t.get('type', None) == "ResultMessage")
        except StopIteration:
            logging.info("Claude exited before returning a result message")
            raise
        except KeyError:
            logging.error("No summary found in result message!")
            raise
        _detect_usage_limit(result)
        problem.metadata['result'] = result
