"""OpenAI Codex SDK model definition.

Ports the :class:`~core.models.claude.Claude` agentic harness to the OpenAI
Codex Python SDK (``openai-codex``). Like the Claude Agent SDK, Codex is a
*complete* coding-agent harness: it spawns the native ``codex`` binary and
drives built-in shell / apply_patch / file tools, an MCP client and a native
OS sandbox over a JSON-RPC app-server. We therefore mirror the Claude wrapper
much more directly than the OpenAI Agents SDK wrapper (which had to hand-wire a
shell tool and a bubblewrap sandbox).

Containment (no contamination between runs) is native here: the Codex thread
runs with ``sandbox="workspace-write"`` and ``approval_mode=deny_all``, so the
agent may only write inside its per-attempt ``cwd`` (the isolated temp env) and
never blocks on interactive approvals. Note workspace-write leaves ``/tmp``
writable by default — the config overlay sets ``exclude_slash_tmp`` /
``exclude_tmpdir_env_var`` so the shared skills dir, env template and sibling
attempt envs under TEMPORARY_DIR are protected. No bubblewrap needed.

NOTE ON SDK SURFACE: the Codex Python SDK is young and its public API
(``Codex`` / ``Thread`` / ``TurnHandle``, the ``config`` overlay for MCP and
model providers, the streaming ``Notification`` payloads) is what this module
targets. If a pinned ``openai-codex`` release renames a field, adjust the thin
adapters below (``_serialize_notification``, ``_extract_message_text``,
``_run_turn``) — the rest of the harness contract is provider-agnostic.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import openai_codex
from openai_codex import (
    Codex,
    CodexConfig as CodexClientConfig,
    Sandbox,
    ApprovalMode,
    TextInput,
    SkillInput,
)

try:  # error surface differs slightly across early releases; degrade gracefully
    from openai_codex import (
        CodexError,
        ServerBusyError,
        is_retryable_error,
    )
except ImportError:  # pragma: no cover - depends on installed SDK version
    CodexError = Exception  # type: ignore[assignment,misc]
    ServerBusyError = Exception  # type: ignore[assignment,misc]

    def is_retryable_error(exc: BaseException) -> bool:  # type: ignore[misc]
        return False

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


CODEX_SDK_VERSION = getattr(openai_codex, "__version__", "unknown")

_CLI_VERSION_CACHE: dict[Optional[str], Optional[str]] = {}


def _codex_cli_version(codex_bin: Optional[str]) -> Optional[str]:
    """Version of the codex runtime that will execute the thread.

    Recorded in each trace because the native-skills discovery scope is
    CLI-version coupled (``<cwd>/.codex/skills`` as of codex-cli 0.145.0;
    newer releases move to ``.agents/skills``): with an unpinned system
    binary, a silent scope mismatch would load no skill content at all,
    indistinguishable from a weak model unless the version is on record.
    """
    if codex_bin not in _CLI_VERSION_CACHE:
        if codex_bin is None:
            try:
                from importlib.metadata import version
                _CLI_VERSION_CACHE[codex_bin] = version('openai-codex-cli-bin')
            except Exception:
                _CLI_VERSION_CACHE[codex_bin] = CODEX_SDK_VERSION
        else:
            try:
                out = subprocess.run([codex_bin, '--version'],
                                     capture_output=True, text=True, timeout=10)
                _CLI_VERSION_CACHE[codex_bin] = out.stdout.strip() or None
            except (OSError, subprocess.TimeoutExpired):
                _CLI_VERSION_CACHE[codex_bin] = None
    return _CLI_VERSION_CACHE[codex_bin]
# Set None to let Codex pick the auth-appropriate default. A ChatGPT-account
# session rejects API-only ids (e.g. "gpt-5.1-codex": "not supported when using
# Codex with a ChatGPT account") — check the supported ids with the `codex` CLI
# /model picker if a run 400s on the model.
CODEX_DEFAULT_MODEL: Optional[str] = "gpt-5.4"

# Shared with the Claude / OpenAI wrappers verbatim so the status scoring is
# identical across providers (comparability of the reported flags).
CODEX_STATUS_PROMPT = {
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

CODEX_USAGE_LIMIT_RETRY_DELAY = 10 * 60  # 10 minutes
# Neither the SDK errors nor the turn-error payloads expose a reset timestamp
# (unlike the Claude SDK's resets_at), so the retry wait is a flat delay.
CODEX_STATUS_TURN_TIMEOUT = 10 * 60  # extra budget for the status turn

# ``codex_error_info`` discriminators (TurnError.codex_error_info) that mark a
# transient, retry-worthy turn failure rather than a genuine bad request —
# usage/overload plus the network/stream-failure variants. Matched
# case/underscore-insensitively (the serialized dict uses snake_case keys).
CODEX_RETRYABLE_ERROR_INFOS = (
    "usageLimitExceeded", "serverOverloaded", "internalServerError",
    "httpConnectionFailed", "responseStreamConnectionFailed",
    "responseStreamDisconnected", "responseTooManyFailedAttempts",
)


def _normalize_error_info(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', text.lower())


def _raise_for_rate_limit(exc: BaseException) -> None:
    """Translate a Codex overload / usage-limit JSON-RPC error into a retry request."""
    if not (isinstance(exc, (CodexError, ServerBusyError)) and is_retryable_error(exc)):
        return
    logging.warning(f"Codex endpoint rate-limited: {exc}")
    raise RetryGenerationException(
        message=f"Codex usage limit hit: '{exc}'",
        wait_seconds=CODEX_USAGE_LIMIT_RETRY_DELAY)


def _raise_for_failed_turn(turn_error: Any) -> None:
    """Raise for a turn that streamed to ``status == failed``.

    In-band turn failures (model 400s, usage limits, server overload) arrive as
    notifications, not exceptions — ``TurnHandle.stream()`` never raises — so
    without this check a failed turn is silently recorded as a successful empty
    run. Retryable failures become :class:`RetryGenerationException`; the rest
    surface as ``RuntimeError`` and are recorded as the problem's exception.
    ``turn_error`` may be ``None``: the SDK marks ``Turn.error`` optional even
    on failure, and a failed turn without details must still not pass as
    success.
    """
    message = getattr(turn_error, 'message', None) or "unknown error"
    info = _to_jsonable(getattr(turn_error, 'codex_error_info', None))
    info_text = _normalize_error_info(str(info))
    if any(_normalize_error_info(kind) in info_text for kind in CODEX_RETRYABLE_ERROR_INFOS):
        logging.warning(f"Codex turn failed with retryable error: {message}")
        raise RetryGenerationException(
            message=f"Codex usage limit / overload: '{message}'",
            wait_seconds=CODEX_USAGE_LIMIT_RETRY_DELAY)
    raise RuntimeError(f"Codex turn failed: {message} (error info: {info})")


@dataclass
class CodexSkillConfig:
    """Configuration for a single skill exposed to the Codex agent."""
    root: str
    subdir: Path | str
    branch: str
    commit: str
    description: str = ""


@dataclass
class CodexMcpServerConfig:
    """Configuration for a single MCP server (stdio transport)."""
    type: str
    command: str | None = None
    args: list[str] | None = None


@dataclass
class ContextConfig:
    """Configuration for the context provided to the Codex agent."""
    src: Optional[Path] = None
    dst: Optional[Path] = Path("task.md")


@dataclass
class CodexConfig(AgenticConfig):
    """Configuration for the Codex agent."""
    model: Optional[str] = CODEX_DEFAULT_MODEL
    # Reasoning depth passed through the Codex config overlay (model_reasoning_effort).
    reasoning_effort: Optional[str] = "high"
    # Native OS sandbox mode: read-only | workspace-write | full-access.
    sandbox: str = "workspace-write"
    # Auth. Default (None) uses the existing `codex login` session
    # (~/.codex/auth.json) — log in once with your ChatGPT subscription via
    # `codex login` (or `codex login --device-code` on a headless host). Set
    # ``api_key_file`` to a file under ``KEYS_DIRECTORY`` to force API-key billing
    # instead (mutually exclusive with subscription auth). Set ``base_url``
    # (+ ``model_provider``) to drive an OpenAI-compatible endpoint.
    api_key_file: Optional[str] = None
    base_url: Optional[str] = None
    model_provider: Optional[str] = None
    # Path to a custom ``codex`` executable. Leave ``None`` to use the binary
    # bundled with ``openai-codex``; set it when that bundled binary has no
    # compatible wheel for the host (e.g. glibc linux, where only musl wheels
    # are published) and a system ``codex`` CLI is installed instead.
    codex_bin: Optional[str] = None
    # True (default): install skills natively by copying each skill dir into
    # the repo discovery scope (``<cwd>/.codex/skills``) and passing
    # ``SkillInput`` turn items referencing the discovered copies. Codex only
    # loads skills from *discovered* scopes ($CODEX_HOME/skills,
    # <cwd>/.codex/skills, ...); a SkillInput pointing at an arbitrary path is
    # accepted but NOT loaded (verified on gpt-5.4), hence the copy.
    # The discovery scope is version-coupled to the system ``codex`` CLI
    # (<cwd>/.codex/skills as of codex-cli 0.145.0; newer releases move to
    # .agents/skills) — re-verify after any CLI update. Slash commands are
    # still expanded by ``_expand_prompt`` in BOTH regimes: Codex has no
    # command concept (custom prompts are TUI-only and deprecated upstream;
    # ``codex exec`` treats "/cmd" as raw text — openai/codex#3641).
    # False (``--no-native``): inline the SKILL.md into the developer
    # instructions instead — the legacy regime of all runs recorded before
    # 2026-08. NOTE: switching regimes changes what the model sees — do not
    # mix within one compared run set.
    native_skills: bool = True
    skills: dict[str, CodexSkillConfig] = field(default_factory=lambda: {
        'lean4-skills': CodexSkillConfig(root="freer-skills", subdir="plugins/lean4",
                                         branch="main", commit="a1cef7dc21d9b68b79311d57cd78ecf6b7ca22bf",
                                         description="Lean 4 proving skills")
    })
    mcp_server: dict[str, CodexMcpServerConfig] = field(default_factory=lambda: {
        "lean-lsp": CodexMcpServerConfig(type="stdio", command="uvx", args=["lean-lsp-mcp"])
    })
    context: Optional[ContextConfig] = None
    prompt: str = ""
    status_prompt: Optional[str] = 'base'
    dependencies: dict[str, str] = field(default_factory=lambda: {
        'openai-codex': CODEX_SDK_VERSION,
        'lean-lsp-mcp': get_lib_version('lean-lsp-mcp'),
        'leanclient': get_lib_version('leanclient'),
    })


def _to_jsonable(obj: Any, depth: int = 0) -> Any:
    """Best-effort recursive conversion to JSON-friendly values."""
    if depth > 6 or obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v, depth + 1) for v in obj]
    dump = getattr(obj, 'model_dump', None)
    if callable(dump):
        # mode='json' keeps enums/dates as their JSON values; python-mode would
        # dump e.g. TurnStatus members as empty dicts in the saved trace.
        try:
            return _to_jsonable(dump(mode='json'), depth + 1)
        except Exception:
            try:
                return _to_jsonable(dump(), depth + 1)
            except Exception:
                pass
    if hasattr(obj, '__dict__'):
        return {k: _to_jsonable(v, depth + 1) for k, v in vars(obj).items()
                if not k.startswith('_')}
    return repr(obj)[:2000]


def _serialize_notification(note: Any) -> dict[str, Any]:
    """Coerce a Codex streaming ``Notification`` into a trace-friendly dict."""
    payload: dict[str, Any] = {'type': type(note).__name__}
    method = getattr(note, 'method', None)
    if method is not None:
        payload['method'] = method
    data = getattr(note, 'payload', None)
    if data is not None:
        dumped = _to_jsonable(data)
        payload['payload'] = {'type': type(data).__name__,
                              **(dumped if isinstance(dumped, dict) else {'value': dumped})}
    return payload


def _extract_message_text(item: Any) -> Optional[str]:
    """Return the assistant text of an agent-message thread item, else ``None``.

    The streamed ``ItemCompletedNotification.item`` is a pydantic ``RootModel``
    union (``ThreadItem``) whose payload lives under ``.root`` — RootModel does
    not proxy attribute access, so the wrapper must be unwrapped first (the
    SDK's own collector does the same in ``openai_codex/_run.py``). The
    ``type`` discriminator (``"agentMessage"``) then skips reasoning/command/
    file items; class names are not stable across the SDK's model modules.
    """
    item = getattr(item, 'root', item)
    if item is None or getattr(item, 'type', None) != 'agentMessage':
        return None
    return getattr(item, 'text', None) or None


def _usage_dict(usage: Any) -> dict[str, Any]:
    """Flatten a ThreadTokenUsage into a plain dict (best effort)."""
    if usage is None:
        return {}
    dumped = _to_jsonable(usage)
    return dumped if isinstance(dumped, dict) else {'value': dumped}


# A slash command as used in the task prompts, e.g. "/lean4:autoprove problem.lean --deep=stuck".
_SLASH_CMD_RE = re.compile(r'^/(?P<plugin>[A-Za-z0-9_-]+):(?P<cmd>[A-Za-z0-9_-]+)\b(?P<rest>.*)$', re.DOTALL)
# A slash command referenced anywhere in a prompt (e.g. the END_TO_END task
# mentions "/lean4:autoprove --deep=stuck" mid-sentence).
_SLASH_CMD_REF_RE = re.compile(r'/(?P<plugin>[A-Za-z0-9_-]+):(?P<cmd>[A-Za-z0-9_-]+)\b')


def _strip_frontmatter(text: str) -> str:
    """Drop a leading YAML ``---`` frontmatter block from a markdown doc."""
    if text.startswith('---'):
        end = text.find('\n---', 3)
        if end != -1:
            nl = text.find('\n', end + 1)
            return text[nl + 1:] if nl != -1 else ''
    return text


class CodexProver(BaseAgentic):
    """Wrapper for the OpenAI Codex SDK theorem prover."""
    PROVER_NAME = "codex"
    CONFIG_CLASS = CodexConfig

    def _read_key(self) -> Optional[str]:
        """Read the API key from ``KEYS_DIRECTORY / api_key_file`` if configured."""
        if self.config.api_key_file is None:
            return None
        key_path = var.KEYS_DIRECTORY / self.config.api_key_file
        if not key_path.exists():
            logging.warning(f"Codex key file '{key_path}' not found; no API key configured")
            return None
        lines = [ln.strip() for ln in key_path.read_text().splitlines() if ln.strip()]
        return lines[-1] if lines else None

    def _resolve_codex_bin(self) -> Optional[str]:
        """Path of the ``codex`` binary to run, or ``None`` for the SDK's bundled one.

        The SDK's bundled runtime lives in the separate ``codex_cli_bin``
        package (openai-codex-cli-bin), which has no compatible wheel on some
        hosts (e.g. glibc linux); fall back to the system ``codex`` CLI then.
        The SDK performs no PATH lookup itself, so the fallback must resolve to
        a concrete path here.
        """
        if self.config.codex_bin is not None:
            return self.config.codex_bin
        if importlib.util.find_spec('codex_cli_bin') is not None:
            return None
        found = shutil.which('codex')
        if found is not None:
            logging.info(f"Bundled Codex runtime not installed; using system binary {found}")
        return found  # None -> the SDK raises its descriptive FileNotFoundError

    def _effective_provider(self) -> Optional[str]:
        """Provider id selected for the turn (``None`` = OpenAI default)."""
        if self.config.base_url is None:
            return self.config.model_provider
        return self.config.model_provider or "custom"

    def _build_config_overlay(self, mcp_env: dict[str, str]) -> dict[str, Any]:
        """Assemble the Codex ``config`` overlay (config.toml equivalent).

        Registers the stdio MCP servers, sets the reasoning effort and, when a
        custom ``base_url`` is configured, defines a model provider so the agent
        talks to an OpenAI-compatible endpoint instead of OpenAI.
        """
        overlay: dict[str, Any] = {}
        if self.config.mcp_server:
            overlay['mcp_servers'] = {
                key: {'command': mcp.command, 'args': mcp.args or [], 'env': mcp_env}
                for key, mcp in self.config.mcp_server.items()
            }
        if self.config.reasoning_effort is not None:
            overlay['model_reasoning_effort'] = self.config.reasoning_effort
        if self.config.sandbox == "workspace-write":
            # workspace-write leaves /tmp and $TMPDIR writable by default, but
            # the shared skills dir, env template and sibling attempt envs all
            # live under TEMPORARY_DIR (/tmp/appv) — exclude them so one agent
            # cannot contaminate later attempts or parallel workers. The cwd
            # (this attempt's env) stays writable even though it is under /tmp.
            overlay['sandbox_workspace_write'] = {
                'exclude_slash_tmp': True, 'exclude_tmpdir_env_var': True}
        if self.config.base_url is not None:
            provider = self._effective_provider()
            overlay['model_providers'] = {
                provider: {
                    'name': provider,
                    'base_url': self.config.base_url,
                    'env_key': "CODEX_PROVIDER_API_KEY",
                    'wire_api': "responses",
                }
            }
        return overlay

    def _skill_root(self, paths: dict[str, Any], skill: CodexSkillConfig) -> Path:
        """On-disk root of a skill's plugin directory (e.g. .../plugins/lean4)."""
        return Path(paths['skills']) / skill.root / skill.subdir

    def _skills_note(self, paths: dict[str, Any]) -> Optional[str]:
        """Inline each skill's SKILL.md so the agent actually picks up the skill.

        Used only when ``native_skills`` is False (``--no-native``): the legacy
        inline regime of the pre-2026-08 recorded runs. Codex had no usable
        skill-discovery channel here, and a bare path advertisement was
        ignored by the model. We inject the skill entry doc (SKILL.md) and point
        at the plugin root, where commands/ and skills/*/references/ can be read
        on demand with the shell tool. Falls back to advertising paths if a skill
        has no SKILL.md.
        """
        docs: list[str] = []
        for key, skill in self.config.skills.items():
            root = self._skill_root(paths, skill)
            skill_mds = sorted(root.glob('skills/*/SKILL.md')) or sorted(root.glob('**/SKILL.md'))
            for skill_md in skill_mds:
                content = _strip_frontmatter(skill_md.read_text()).strip()
                docs.append(
                    f"# Skill '{key}': {skill.description}\n"
                    f"Command definitions (`commands/*.md`) and reference docs "
                    f"(`skills/*/references/*.md`) live under `{root}` — read them with the "
                    f"shell tool as needed. Skill entry follows:\n\n{content}")
        if docs:
            return "\n\n".join(docs)
        lines = [f"- {key}: {skill.description} (files under {self._skill_root(paths, skill)})"
                 for key, skill in self.config.skills.items()]
        if not lines:
            return None
        return ("Available skills — read their files with the shell tool to use them:\n"
                + "\n".join(lines))

    def _find_command_file(self, paths: dict[str, Any], plugin: str, cmd: str) -> Optional[Path]:
        """Locate ``commands/<cmd>.md`` in the skill whose plugin dir is ``plugin``.

        The plugin segment of ``/plugin:cmd`` is matched against the skill
        root's directory name (e.g. ``plugins/lean4`` -> ``lean4``) so that a
        ``/other:autoprove`` reference does not silently expand to lean4's
        command file.
        """
        for skill in self.config.skills.values():
            root = self._skill_root(paths, skill)
            if root.name != plugin:
                continue
            cand = root / 'commands' / f'{cmd}.md'
            if cand.is_file():
                return cand
        return None

    def _expand_prompt(self, prompt: str, paths: dict[str, Any]) -> str:
        """Expand ``/plugin:cmd`` slash commands into their command bodies.

        The task prompts use Claude plugin commands (e.g. ``/lean4:autoprove``)
        which are opaque strings to Codex in the inline-skills regime. A prompt
        that *is* a slash command is rewritten into the command's markdown
        definition; a prompt that merely *references* commands mid-text (e.g.
        the END_TO_END task) gets each referenced command's definition appended.
        Non-command prompts and the ``--no-skills`` case (empty ``skills``)
        pass through unchanged. Applied in both skill regimes: Codex's native
        skills cover SKILL.md only — it has no command concept, so ``/plugin:cmd``
        strings always need expanding here.
        """
        m = _SLASH_CMD_RE.match(prompt.strip())
        if m is not None:
            plugin, cmd, rest = m.group('plugin'), m.group('cmd'), m.group('rest').strip()
            cmd_file = self._find_command_file(paths, plugin, cmd)
            if cmd_file is None:
                return prompt
            body = _strip_frontmatter(cmd_file.read_text()).strip()
            root = cmd_file.parent.parent
            return (
                f"Execute the `/{plugin}:{cmd}` command with arguments: `{rest}`.\n"
                f"Its definition is below. Paths it references (e.g. `../skills/...`, "
                f"`references/...`) are relative to the command file `{cmd_file}` "
                f"(plugin root `{root}`) — read them with the shell tool as needed.\n\n"
                f"--- BEGIN COMMAND /{plugin}:{cmd} ---\n{body}\n--- END COMMAND /{plugin}:{cmd} ---")
        # Mid-text references: append each referenced command's definition.
        blocks: list[str] = []
        seen: set[tuple[str, str]] = set()
        for ref in _SLASH_CMD_REF_RE.finditer(prompt):
            plugin, cmd = ref.group('plugin'), ref.group('cmd')
            if (plugin, cmd) in seen:
                continue
            seen.add((plugin, cmd))
            cmd_file = self._find_command_file(paths, plugin, cmd)
            if cmd_file is None:
                continue
            body = _strip_frontmatter(cmd_file.read_text()).strip()
            blocks.append(
                f"--- BEGIN COMMAND /{plugin}:{cmd} (file `{cmd_file}`) ---\n"
                f"{body}\n--- END COMMAND /{plugin}:{cmd} ---")
        if not blocks:
            return prompt
        return (
            f"{prompt}\n\n"
            "The slash commands referenced above are defined below. Paths a "
            "definition references (e.g. `../skills/...`, `references/...`) are "
            "relative to its command file — read them with the shell tool as "
            "needed.\n\n" + "\n\n".join(blocks))

    def _skill_env(self, paths: dict[str, Any]) -> dict[str, str]:
        """Env vars each plugin's SessionStart bootstrap hook would export.

        Claude Code runs ``hooks/bootstrap.sh`` at session start, which exports
        e.g. ``LEAN4_SCRIPTS`` — the inlined SKILL.md invokes its scripted
        primitives through these variables. Outside Claude Code the hook never
        runs, so execute it here (with ``CLAUDE_PLUGIN_ROOT``/``CLAUDE_ENV_FILE``
        set the way Claude Code would) and collect the exported variables.
        """
        env: dict[str, str] = {}
        for key, skill in self.config.skills.items():
            root = self._skill_root(paths, skill)
            hook = root / 'hooks' / 'bootstrap.sh'
            if not hook.is_file():
                continue
            with tempfile.NamedTemporaryFile('r', suffix='.env') as env_file:
                proc = subprocess.run(
                    ['bash', str(hook)], capture_output=True, text=True, timeout=30,
                    env={**os.environ, 'CLAUDE_PLUGIN_ROOT': str(root),
                         'CLAUDE_ENV_FILE': env_file.name})
                if proc.returncode != 0:
                    logging.warning(f"Skill '{key}' bootstrap hook failed: {proc.stderr.strip()}")
                    continue
                for line in Path(env_file.name).read_text().splitlines():
                    line = line.strip()
                    if not line.startswith('export '):
                        continue
                    name, _, value = line[len('export '):].partition('=')
                    parts = shlex.split(value)
                    env[name.strip()] = parts[0] if parts else ''
        return env

    def _install_native_skills(self, paths: dict[str, Any]) -> list[Any]:
        """Install skills into the repo discovery scope and return ``SkillInput`` items.

        Codex only loads skills it has itself discovered (scopes:
        ``$CODEX_HOME/skills``, ``<cwd>/.codex/skills``, ...) — a ``SkillInput``
        pointing at an arbitrary path is accepted but not loaded. Copy each
        skill dir (the directory containing SKILL.md, with its references/)
        into ``<cwd>/.codex/skills/<name>`` so discovery picks it up, then
        reference the discovered SKILL.md from the turn input.
        """
        inputs: list[Any] = []
        dest_root = Path(paths['env']) / '.codex' / 'skills'
        for skill in self.config.skills.values():
            root = self._skill_root(paths, skill)
            skill_mds = sorted(root.glob('skills/*/SKILL.md')) or sorted(root.glob('**/SKILL.md'))
            for skill_md in skill_mds:
                name = skill_md.parent.name
                dst = dest_root / name
                shutil.copytree(skill_md.parent, dst, dirs_exist_ok=True)
                inputs.append(SkillInput(name=name, path=str(dst / 'SKILL.md')))
        return inputs

    def _run_turn(self, thread: Any, problem: Problem, turn_input: Any,
                  trace: list[dict[str, Any]], timeout: float) -> tuple[Optional[str], Any, Optional[str]]:
        """Run one Codex turn, streaming notifications into ``trace``.

        ``turn_input`` is any Codex ``RunInput`` (a prompt string, or a list of
        input items such as ``TextInput`` + ``SkillInput``). ``timeout`` is the
        elapsed-seconds ceiling for this turn (checked against
        ``problem.get_elapsed()``). Returns ``(final_text, usage,
        timeout_status)`` where ``timeout_status`` is ``"TIMEOUT"`` if the
        ceiling was hit (and the turn was interrupted), else ``None``.

        Raises for turns that stream to ``status == failed``: the stream itself
        never raises on in-band errors (model 400s, usage limits), so the
        terminal ``turn/completed`` notification is inspected explicitly.
        """
        handle = thread.turn(turn_input)
        final_text: Optional[str] = None
        usage: Any = None
        timeout_status: Optional[str] = None
        turn_failed = False
        turn_error: Any = None
        for note in handle.stream():
            trace.append(_serialize_notification(note))
            logging.debug(f"New Codex notification: {trace[-1]}")
            data = getattr(note, 'payload', None)
            method = getattr(note, 'method', None)
            if method == 'error':
                message = getattr(getattr(data, 'error', None), 'message', None)
                if getattr(data, 'will_retry', False):
                    logging.warning(f"Codex turn error (server retrying): {message}")
                else:
                    logging.error(f"Codex turn error: {message}")
            elif data is not None:
                if method == 'turn/completed':
                    turn = getattr(data, 'turn', None)
                    status_val = getattr(turn, 'status', None)
                    if getattr(status_val, 'value', status_val) == 'failed':
                        # Turn.error is optional even on failure — track the
                        # failed status itself so an error-less failure still
                        # raises instead of passing as an empty success.
                        turn_failed = True
                        turn_error = getattr(turn, 'error', None)
                text_item = _extract_message_text(getattr(data, 'item', None))
                if text_item is not None:
                    final_text = text_item
                token_usage = getattr(data, 'token_usage', None)
                if token_usage is not None:
                    usage = token_usage
            # A failure observed in the same iteration takes precedence over the
            # deadline: masking it as TIMEOUT would suppress the retry.
            if timeout_status != "TIMEOUT" and not turn_failed and problem.get_elapsed() > timeout:
                try:
                    handle.interrupt()
                except Exception as exc:  # pragma: no cover - best-effort interrupt
                    logging.debug(f"Codex interrupt failed: {exc}")
                logging.warning(f"Codex turn timed out after {timeout} seconds.")
                timeout_status = "TIMEOUT"
                break
        if turn_failed and timeout_status is None:
            _raise_for_failed_turn(turn_error)
        return final_text, usage, timeout_status

    def codex(self, problem: Problem, prompt: str, paths: dict[str, Path]) -> tuple[
        list[dict[str, Any]], Optional[str]]:
        """Run the Codex agent with a native workspace sandbox and stream the response."""
        if self.config.context is not None:
            src = var.CONTEXTS_DIRECTORY / self.config.context.src
            context_text = src.read_text()
            if self.config.context.dst is not None:
                (paths['env'] / self.config.context.dst).write_text(context_text)
        else:
            context_text = None
        # Native skills: install into the repo discovery scope and attach
        # SkillInput turn items. Fallback: inline SKILL.md into the developer
        # instructions. The task context always goes through the *additive*
        # developer_instructions channel: base_instructions would REPLACE
        # Codex's built-in agent prompt (tool-use/sandbox conventions), whereas
        # claude.py appends the same context to its preset system prompt.
        skill_inputs = self._install_native_skills(paths) if self.config.native_skills else []
        skills_note = None if self.config.native_skills else self._skills_note(paths)
        developer_instructions = "\n\n".join(
            part for part in (context_text, skills_note) if part) or None

        # Environment for both the codex subprocess and the stdio MCP servers.
        # Keep PATH/ELAN so `uvx lean-lsp-mcp` and `lake`/`elan` resolve.
        run_env = dict(os.environ)
        run_env.setdefault('LEAN_LOG_LEVEL', 'WARNING')
        run_env.update(self._skill_env(paths))
        api_key = self._read_key()
        if self.config.base_url is not None:
            if api_key is not None:
                run_env['CODEX_PROVIDER_API_KEY'] = api_key
            elif 'CODEX_PROVIDER_API_KEY' not in run_env:
                logging.warning(
                    "base_url set but no API key available (api_key_file missing/unset "
                    "and CODEX_PROVIDER_API_KEY not exported); the model provider will "
                    "likely reject requests")
        elif api_key is not None:
            # API-key auth persists the account session under CODEX_HOME. Scope
            # it to the attempt env so the machine-global ~/.codex (e.g. a
            # ChatGPT-subscription login) is never overwritten and concurrent
            # runs with different keys cannot race on shared auth state.
            codex_home = Path(paths['env']) / '.codex-home'
            codex_home.mkdir(parents=True, exist_ok=True)
            run_env['CODEX_HOME'] = str(codex_home)

        client_config = CodexClientConfig(env=run_env, codex_bin=self._resolve_codex_bin())
        trace: list[dict[str, Any]] = list()
        status: Optional[str] = None
        problem.reset_timer()

        with Codex(config=client_config) as codex_client:
            if api_key is not None and self.config.base_url is None:
                # Default OpenAI endpoint: authenticate with the stored key
                # (written to the per-attempt CODEX_HOME set above).
                codex_client.login_api_key(api_key)

            thread = codex_client.thread_start(
                model=self.config.model,
                cwd=str(paths['env']),
                sandbox=Sandbox(self.config.sandbox),
                approval_mode=ApprovalMode.deny_all,
                developer_instructions=developer_instructions,
                model_provider=self._effective_provider(),
                config=self._build_config_overlay(run_env),
            )

            main_input = [TextInput(text=prompt), *skill_inputs] if skill_inputs else prompt
            try:
                final_text, usage, timeout_status = self._run_turn(
                    thread, problem, main_input, trace, timeout=self.config.generation_timeout)
            except (CodexError, ServerBusyError) as exc:
                _raise_for_rate_limit(exc)
                raise
            if timeout_status is not None:
                status = timeout_status
            trace.append({'type': 'ResultMessage',
                          'result': final_text if final_text is not None else "",
                          'model': self.config.model,
                          'usage': _usage_dict(usage),
                          'codex_cli_version': _codex_cli_version(self._resolve_codex_bin()),
                          'native_skills': self.config.native_skills,
                          'total_cost_usd': None})  # Codex reports no cost

            if self.config.status_prompt is not None and status != "TIMEOUT":
                # The status turn gets its own small budget on top of whatever
                # the main turn spent, so a near-timeout run still yields a flag.
                try:
                    status_text, status_usage, status_timeout = self._run_turn(
                        thread, problem, CODEX_STATUS_PROMPT[self.config.status_prompt], trace,
                        timeout=problem.get_elapsed() + CODEX_STATUS_TURN_TIMEOUT)
                except (CodexError, ServerBusyError) as exc:
                    _raise_for_rate_limit(exc)
                    raise
                except RuntimeError as exc:
                    # A hard-failed *status* turn must not discard the completed
                    # main turn's trace and result (claude.py parity: a bad
                    # status reply just leaves status None). Retryable failures
                    # (RetryGenerationException) still propagate and re-run the
                    # attempt, as the Claude wrapper does on its status query.
                    logging.error(f"Codex status turn failed; recording status None: {exc}")
                    status_text, status_usage, status_timeout = None, None, None
                if status is None:
                    # Strip markdown emphasis some models wrap the flag in
                    # ("**ISSUE**" -> "ISSUE") so consumers match exact strings.
                    status = (status_text.strip().strip('*_#').strip()
                              if status_text is not None else status_timeout)
                trace.append({'type': 'StatusMessage', 'status': status,
                              'usage': _usage_dict(status_usage)})

        return trace, status

    def _agent(self, problem: Problem, paths: dict[str, Path | list[Path]] | None = None, **kwargs: Any) -> None:
        """Build the prompt and invoke the Codex agent."""
        prompt = self.config.prompt.format(filepath=" ".join(s.name for s in paths['problems']))
        prompt = self._expand_prompt(prompt, paths)
        try:
            trace, status = self.codex(problem=problem, prompt=prompt, paths=paths)
        except (CodexError, ServerBusyError) as exc:
            _raise_for_rate_limit(exc)
            raise
        # Keyed on the bare prover name (like claude.py's literal 'claude'):
        # consumers (clever.py summaries, display.py) look the trace up under
        # the -p prover id, which never carries the name_modifier suffix.
        problem.metadata.update({'prompt': prompt, self.PROVER_NAME: trace, 'status': status})
        try:
            result = next(t['result'] for t in trace if t.get('type', None) == "ResultMessage")
        except StopIteration:
            logging.info("Codex exited before returning a result message")
            raise
        except KeyError:
            logging.error("No summary found in result message!")
            raise
        problem.metadata['result'] = result


### GPT-5.1 — contamination-control variant

@dataclass
class CodexPreConfig(CodexConfig):
    # GPT-5.1-Codex-Max: the strongest agentic model of the newest OpenAI
    # generation whose stated knowledge cutoff (2024-09-30, all gpt-5.1
    # variants) precedes CLEVER's public release (arXiv v1 2025-05-20) — the
    # contamination-control column. The -codex-max variant was the Codex
    # CLI's own default model of its era, so harness fit is native; swap to
    # plain "gpt-5.1" for strict variant-class parity with the main
    # `-p codex` (plain gpt-5.4) runs. Everything else inherits CodexConfig.
    # API-key billing (agents/keys/openai): the ChatGPT-subscription
    # allowlist carries only current models, so the retired 5.1-era ids are
    # API-only. Auth is scoped to a per-attempt CODEX_HOME — the global
    # ~/.codex subscription login used by `-p codex` is never touched.
    model: Optional[str] = "gpt-5.1"
    api_key_file: Optional[str] = "openai"


class CodexPre(CodexProver):
    """Codex agent pinned to GPT-5.1 (pre-CLEVER training cutoff).

    Distinct ``PROVER_NAME`` so records, traces and summaries land under
    ``data/codex-pre`` and the ``codex-pre`` metadata key — fully separate
    from the ``data/codex`` (gpt-5.4) run set.
    """
    PROVER_NAME = "codex-pre"
    CONFIG_CLASS = CodexPreConfig
