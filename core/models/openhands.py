"""OpenHands SDK model definition.

Ports the :class:`~core.models.claude.Claude` agentic harness to the OpenHands
``software-agent-sdk``. OpenHands runs fully headless in-process: a
``LocalConversation`` drives litellm-backed LLM calls and executes built-in
tools (bash via ``TerminalTool``, file edits via ``FileEditorTool``) directly on
the host, rooted at the per-attempt workspace directory. This is used to test
open-weight models (Kimi / MiniMax / GLM) served by an institutional
OpenAI-compatible endpoint, mirroring the OpenCode provider.

CONTAINMENT CAVEAT: unlike the Claude/Codex wrappers (which get a native OS
sandbox) the local OpenHands runtime executes tools on the host with no
filesystem allowlist — isolation here rests only on the fresh temp workspace
that :class:`BaseAgentic` creates per attempt. In particular the workspace's
``.lake/packages`` symlinks (copied with ``symlinks=True``) point into the
shared Lean cache under ``/var/cache/lean``, so a stray agent write there would
leak across runs (the Claude wrapper's permission handler does not path-check
Bash either, so this exposure is shared, not a regression). OpenHands' own sandbox is
Docker-based (``openhands.workspace.DockerWorkspace`` + the
``ghcr.io/openhands/agent-server`` image); wiring that in is left as a follow-up
(see ``_make_workspace``) so that fidelity to the Claude pipeline comes first.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pydantic import SecretStr, model_validator

# Silence the SDK's startup banner and litellm's chatty logging *before* the
# openhands import runs (the banner prints at import time).
os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
os.environ.setdefault("LITELLM_LOG", "ERROR")

import openhands.sdk as openhands_sdk
from openhands.sdk import LLM, Agent, AgentContext, Conversation, Tool, Event, LLMConvertibleEvent
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.mcp import MCPServer
from openhands.sdk.plugin import PluginSource
from openhands.tools.terminal import TerminalTool
from openhands.tools.file_editor import FileEditorTool

try:  # exception surface lives under a private-ish path; degrade gracefully
    from openhands.sdk.llm.exceptions.types import LLMRateLimitError
except ImportError:  # pragma: no cover - depends on installed SDK version
    class LLMRateLimitError(Exception):  # type: ignore[no-redef]
        pass

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


# Namespaces that spam INFO chatter during a run (OpenHands tool/pane setup, the
# MCP client's per-request lines, litellm's per-call HTTP logs).
_NOISY_LOG_PREFIXES = ("openhands", "mcp", "litellm", "LiteLLM", "httpx", "httpcore")


def _quiet_dependencies() -> None:
    """Mute third-party INFO logs and cost warnings unless the root is at DEBUG.

    Runs inside each worker (post-fork, so it sees the configured level), before
    the agent/conversation are built. OpenHands, its MCP client and litellm emit
    a lot of INFO chatter plus a per-call "Cost calculation failed" UserWarning
    for unpriced models (the institutional endpoint) — all of it is hidden below DEBUG.
    """
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        return
    names = set(_NOISY_LOG_PREFIXES) | {
        n for n in logging.root.manager.loggerDict
        if any(n == p or n.startswith(p + ".") for p in _NOISY_LOG_PREFIXES)
    }
    for name in names:
        logging.getLogger(name).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message="Cost calculation failed")
    # The lean4 plugin's hooks.json is Claude-Code format with a benign extra
    # "description" key next to "hooks"; the SDK parses it fine but warns once
    # per plugin load (= once per problem). Only that metadata key is ignored,
    # and the hooks are inert in the SDK anyway (see _skill_env), so drop it.
    logging.getLogger("openhands.sdk.hooks.config").addFilter(
        lambda record: "legacy wrapper format" not in record.getMessage())


OPENHANDS_SDK_VERSION = getattr(openhands_sdk, "__version__", "unknown")
# Endpoint and key file are site-specific: set APPV_OPENHANDS_BASE_URL (or put
# the base URL on the first line of the key file) and provide the key file.
OPENHANDS_DEFAULT_BASE_URL = os.environ.get("APPV_OPENHANDS_BASE_URL", "https://<endpoint>/api")
OPENHANDS_DEFAULT_KEY_FILE = "openhands"
OPENHANDS_DEFAULT_MODEL = "openai/Kimi"

# Shared verbatim with the Claude / OpenAI wrappers so status scoring is
# identical across providers (comparability of the reported flags).
OPENHANDS_STATUS_PROMPT = {
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

# Corrective prompts for the in-run section validation (Checks A/B/C in
# :meth:`OpenHands.openhands`); ``str.format`` templates, kept next to the
# status prompts so all model-facing wording lives in one place.
OPENHANDS_VALIDATION_PROMPT = {
    'tags': "Your solution removed or renamed required section tags: {missing} in "
            "`{filename}`. Restore the exact tags `-- [NAME]` in their correct "
            "positions, keeping your work, so every section of the original file "
            "is present.",
    'fixed': "The sections {changed} must not be modified. I have restored their "
             "original content in `{filename}`. Review the restored file: adapt "
             "ONLY the editable sections so the file still compiles and the task "
             "is completed, without touching the restored sections. If the task "
             "is impossible without modifying them, do not modify them and state your impossibility argument.",
    'axiom': "Remove the added axiom(s) {axioms}; axioms are forbidden.",
    'preamble': "The content before the first section tag (imports and given "
                "definitions) must not be modified. I have restored it in "
                "`{filename}`. Move any helper definitions you added there into "
                "the appropriate editable section instead — content outside the "
                "tagged sections is discarded during evaluation.",
    'redecl': "The editable section(s) re-declare existing definitions: "
              "{redecl}. A body section continues the fixed signature above it "
              "— write only the body expression there (no `def ... :=` header), "
              "and never re-declare a given definition.",
    'flag': "\nReply with EXACTLY one word among OK, FAIL, MISMATCH, ISSUE.",
}

OPENHANDS_USAGE_LIMIT_RETRY_DELAY = 10 * 60  # 10 minutes
# "Disabled" sentinels for SDK caps that require an int value.
OPENHANDS_NO_ITERATION_LIMIT = 1_000_000
OPENHANDS_NO_EVENT_CAP = 1_000_000
# Neither the SDK nor the endpoint exposes a reset timestamp (unlike the Claude
# SDK's resets_at), so the retry wait is a flat delay rather than adaptive.


def _raise_for_rate_limit(exc: BaseException) -> None:
    """Translate a (possibly wrapped) rate-limit error into a retry request.

    ``Conversation.run()`` wraps everything raised inside the run loop in a
    ``ConversationRunError``, so a plain ``isinstance`` check on the caught
    exception never sees the ``LLMRateLimitError`` — walk the wrapper's
    ``original_exception`` and the ``__cause__``/``__context__`` chain instead.
    """
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, LLMRateLimitError):
            logging.warning(f"OpenHands endpoint rate-limited: {cur}")
            raise RetryGenerationException(
                message=f"OpenHands usage limit hit: '{cur}'",
                wait_seconds=OPENHANDS_USAGE_LIMIT_RETRY_DELAY)
        cur = (getattr(cur, 'original_exception', None)
               or cur.__cause__ or cur.__context__)


# Exception *type names* and message substrings that indicate a transient,
# server-side failure worth re-running (as opposed to a genuine bad request or a
# bug). Matched across the whole __cause__/__context__ chain so it also catches
# a litellm error wrapped in OpenHands' ConversationRunError. Phrases are kept
# specific: a real malformed request ("context length exceeded", "invalid model")
# won't match and is surfaced as a real failure instead of looping forever.
_TRANSIENT_EXC_NAMES = (
    "InternalServerError", "ServiceUnavailableError", "APIConnectionError",
    "APITimeoutError", "Timeout", "APIError", "ConnectError", "ReadTimeout",
    "RemoteProtocolError",
)
_TRANSIENT_MSG_SUBSTRINGS = (
    "server connection error", "open webui", "connection error", "bad gateway",
    "service unavailable", "internal server error", "temporarily unavailable",
    "overloaded", "connection reset", "connection aborted", "connection refused",
    " 502", " 503", " 504", "econnreset",
)


def _is_transient_error(exc: BaseException) -> bool:
    """True if ``exc`` (or a wrapped cause) looks like a transient endpoint blip."""
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if type(cur).__name__ in _TRANSIENT_EXC_NAMES:
            return True
        msg = str(cur).lower()
        if any(s in msg for s in _TRANSIENT_MSG_SUBSTRINGS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


# Env var to override DNS resolution without sudo/etc/hosts, e.g.
#   APPV_HOST_RESOLVE="endpoint.example.org=192.0.2.10"
# (comma-separated for multiple). Keeps the hostname in the URL so TLS/SNI/cert
# still validate, but connects to the given IP — the client-side `curl --resolve`.
_HOST_RESOLVE_ENV = "APPV_HOST_RESOLVE"


def _parse_host_overrides(spec: Optional[str]) -> dict[str, str]:
    """Parse ``host=ip,host2=ip2`` into a mapping (ignores malformed entries)."""
    out: dict[str, str] = {}
    for pair in (spec or "").split(","):
        pair = pair.strip()
        if "=" in pair:
            host, ip = pair.split("=", 1)
            if host.strip() and ip.strip():
                out[host.strip()] = ip.strip()
    return out


def _install_host_overrides(overrides: dict[str, str]) -> None:
    """Redirect ``socket.getaddrinfo`` for the given hostnames to fixed IPs.

    Applied per worker process. httpx/litellm still send the original hostname as
    SNI and Host and validate the certificate against it (the IP is only used for
    the TCP connect), so this is the no-sudo equivalent of a ``/etc/hosts`` entry
    or ``curl --resolve`` — fixing "Connection error" when DNS can't resolve the
    host but the IP is reachable and the cert is issued for the hostname.
    """
    if not overrides:
        return
    import socket
    store = getattr(socket, "_appv_host_overrides", None)
    if store is None:
        store = {}
        _orig = socket.getaddrinfo

        def _patched(host, *args, **kwargs):
            return _orig(store.get(host, host), *args, **kwargs)

        socket.getaddrinfo = _patched
        socket._appv_host_overrides = store  # type: ignore[attr-defined]
    store.update(overrides)


@dataclass
class OpenHandsSkillConfig:
    """Configuration for a single skill advertised to the OpenHands agent."""
    root: str
    subdir: Path | str
    branch: str
    commit: str
    description: str = ""


@dataclass
class OpenHandsMcpServerConfig:
    """Configuration for a single MCP server (stdio transport)."""
    type: str
    command: str | None = None
    args: list[str] | None = None


@dataclass
class ContextConfig:
    """Configuration for the context provided to the OpenHands agent."""
    src: Optional[Path] = None
    dst: Optional[Path] = Path("task.md")


@dataclass
class OpenHandsConfig(AgenticConfig):
    """Configuration for the OpenHands agent (OpenAI-compatible endpoint)."""
    # litellm-style provider-prefixed model id (e.g. ``openai/Kimi``).
    model: str = OPENHANDS_DEFAULT_MODEL
    base_url: Optional[str] = OPENHANDS_DEFAULT_BASE_URL
    # API key read from ``KEYS_DIRECTORY / api_key_file`` (two-line file may also
    # override the base URL on its first line, single-line file is the key).
    api_key_file: str = OPENHANDS_DEFAULT_KEY_FILE
    reasoning_effort: Optional[str] = None
    # None (default): no iteration cap — the wall-clock generation_timeout is
    # the only budget, matching the Claude wrapper (which has no turn cap
    # either; kimi SPEC_ISO/036 was cut by the old 500 cap 44 min before its
    # 4h wall). The SDK requires an int, so None maps to a huge sentinel.
    max_turns: Optional[int] = None
    # Re-run the whole attempt in a fresh env on a transient endpoint failure
    # (e.g. "Open WebUI: Server Connection Error", 5xx, connection resets).
    transient_retry_delay: int = 60      # seconds to wait before re-running
    max_transient_retries: int = 5       # give up (record the error) after this many
    # Hostname -> IP overrides (no-sudo /etc/hosts alternative). Merged with the
    # APPV_HOST_RESOLVE env var. Use when DNS can't resolve the endpoint but the
    # IP is reachable: keep the hostname in base_url so TLS/cert still validate.
    dns_overrides: dict[str, str] = field(default_factory=dict)
    # Slow open-weight endpoints need a larger wall than the 1h base default;
    # the condenser (below) keeps per-turn latency flat so this is safe headroom.
    generation_timeout: Optional[float] = 60 * 60 * 4
    # --- Context condensation (off for now: isolate the caching variable first) ---
    # Without a condenser, history grows unbounded across ~hundreds of events and
    # each turn re-prefills the whole conversation, so turns slow toward the wall.
    # LLMSummarizingCondenser bounds the view (max_tokens hard trigger ~half the
    # served context window; max_size secondary event cap, keep_first < max_size//2).
    # Enabled: unbounded history otherwise overflows the model's context window
    # (Kimi 256k) and the endpoint hard-fails with ContextWindowExceededError.
    condenser_enabled: bool = True
    # Hard condensation trigger (tokens). If None, derived from the model's
    # context window as ``condenser_context_fraction * max_input_tokens``. The
    # fraction is only an efficiency margin, not a safety one: OpenHands also
    # condenses *reactively* when the endpoint returns a context-window error
    # (agent catches LLMContextWindowExceedError -> condense + retry), so an
    # occasional overflow from approximate token counting is recovered, not
    # fatal. A lower fraction just avoids those wasted overflow round-trips.
    # NOTE: the window covers prompt+completion, so 1-fraction is also the
    # model's *output* headroom before the reactive path kicks in — 0.99 of a
    # 256k window leaves ~2.6k tokens per reply; drop to ~0.95 if truncated
    # completions / overflow retries show up. On the TOKENS trigger the SDK
    # condenses down to max_tokens // 2. Explicit value overrides.
    condenser_max_tokens: Optional[int] = None
    condenser_context_fraction: float = 0.99
    # Event-count trigger (SDK max_size). None (default): disabled — condense
    # on the token trigger only, so summarization (and its context loss) only
    # happens when the window is actually full. The old default of 120 fired
    # every ~65 events while the view sat at ~40k/259k tokens (14 needless
    # condensations in kimi SPEC_ISO/036). The SDK requires an int, so None
    # maps to a huge sentinel. Set an int to restore an event cap.
    condenser_max_size: Optional[int] = None
    condenser_keep_first: int = 4
    # Served context window (tokens). Set per model so the condenser threshold and
    # OpenHands' own context math are correct (Kimi = 262144). None -> unknown,
    # condenser falls back to a conservative absolute cap.
    max_input_tokens: Optional[int] = None
    # --- LLM latency / determinism knobs ---
    # temperature / max_output_tokens default to None = leave the endpoint/model
    # default untouched, matching the Claude wrapper (which sets neither) for
    # cross-backend comparability. Set them explicitly only for a specific run.
    temperature: Optional[float] = None       # None -> endpoint default (parity with Claude)
    seed: Optional[int] = None                # vLLM sampling reproducibility (opt-in)
    max_output_tokens: Optional[int] = None    # None -> endpoint default; do not cap blindly
    http_timeout: int = 900                   # per-request ceiling > default 300s (avoids retry storms)
    # Stream completions (stream=True + a no-op token sink). Required for the
    # institutional endpoint: its Open WebUI proxy injects
    # stream_options={"include_usage": True} into the upstream vLLM request to
    # capture token usage, and a vLLM upgrade now rejects that param unless
    # stream=True is also present ("Stream options can only be defined when
    # stream=True"). Our client never sends stream_options itself, so the only
    # lever is to make the request genuinely streaming so stream=True accompanies
    # the injected option. Bonus: the SDK only asks for detailed usage
    # (prompt_tokens_details.cached_tokens) on the streaming path, so this also
    # makes report_cached_tokens actually deliver. The SDK falls back to
    # non-streaming unless a token callback is wired, so this flag drives both.
    stream: bool = True
    # Ask an OpenAI-compatible server to report prefix-cache hits in
    # usage.prompt_tokens_details.cached_tokens (vLLM/SGLang; no-op if unsupported).
    report_cached_tokens: bool = True
    # Speculative: suppress hybrid-reasoning "thinking" via the served chat
    # template (chat_template_kwargs.enable_thinking=False). Off by default —
    # the flag name is model/template specific; enable per-model after testing.
    disable_thinking: bool = False
    # Opt-in MCP tool allow-list (anchored regex). Left None because it is
    # version-coupled to lean-lsp-mcp's tool names: a wrong/stale regex would
    # filter out ALL lean tools and cripple the agent. Pin lean-lsp-mcp first.
    tool_filter_regex: Optional[str] = None
    # True (default): load the plugin natively via
    # ``Conversation(plugins=[PluginSource(...)])`` — the SDK reads the
    # Claude-Code plugin structure (.claude-plugin/plugin.json, skills/,
    # commands/) itself, lists skills with progressive disclosure
    # (<available_skills> + invoke_skill tool) and registers each command as a
    # skill keyword-triggered by the literal ``/lean4:autoprove``-style string,
    # so no manual inlining/expansion is needed (verified end-to-end on SDK
    # 1.36.1: trigger fires on both the whole-prompt and mid-sentence quoted
    # forms, invoke_skill auto-attaches, plugin skills merge with the wrapper's
    # AgentContext without clobbering the system suffix). Plugin hooks stay
    # inert in the SDK either way; ``_skill_env`` compensates in both regimes.
    # False (``--no-native``): inline each skill's SKILL.md into the system
    # prompt suffix and expand ``/plugin:cmd`` prompts into the command body —
    # the legacy regime of all runs recorded before 2026-08. NOTE: switching
    # regimes changes what the model sees — do not mix within one compared
    # run set.
    native_skills: bool = True
    skills: dict[str, OpenHandsSkillConfig] = field(default_factory=lambda: {
        'lean4-skills': OpenHandsSkillConfig(root="freer-skills", subdir="plugins/lean4",
                                             branch="main", commit="a1cef7dc21d9b68b79311d57cd78ecf6b7ca22bf",
                                             description="Lean 4 proving skills")
    })
    mcp_server: dict[str, OpenHandsMcpServerConfig] = field(default_factory=lambda: {
        "lean-lsp": OpenHandsMcpServerConfig(type="stdio", command="uvx", args=["lean-lsp-mcp"])
    })
    context: Optional[ContextConfig] = None
    prompt: str = ""
    status_prompt: Optional[str] = 'base'
    # --- In-run solution validation (section guards) ---
    # Maps section label -> 'fixed' | 'editable'; set by scripts/clever.py from
    # the task's BlockSpecs (hidden = agent-authored = editable). None/empty
    # disables Checks A/B/C in openhands(), leaving other benchmarks/regimes
    # unaffected. validation_max_retries is the corrective-round budget PER
    # CHECK (A, B and C each get their own count, per attempt).
    validation_sections: Optional[dict[str, str]] = None
    validation_max_retries: int = 3
    dependencies: dict[str, str] = field(default_factory=lambda: {
        'openhands-sdk': OPENHANDS_SDK_VERSION,
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


def _serialize_event(event: Any) -> dict[str, Any]:
    """Coerce an OpenHands ``Event`` into a trace-friendly dict."""
    payload = _to_jsonable(event)
    if not isinstance(payload, dict):
        payload = {'value': payload}
    return {'type': type(event).__name__, **payload}


def _assistant_text(event: Any) -> Optional[str]:
    """Return the agent's final communication carried by an event, else ``None``.

    Two shapes carry the agent's answer: a plain assistant ``MessageEvent`` (chat
    reply, e.g. the status flag), or — when the agent ends by calling the
    ``finish`` tool without a trailing chat message — the ``message`` argument of
    that finish ``ActionEvent``. The built-in system prompt is emitted as a
    ``SystemPromptEvent`` (also ``source='agent'``) and must be excluded.
    """
    name = type(event).__name__
    if name == "ActionEvent" and getattr(event, 'tool_name', None) == "finish":
        action = getattr(event, 'action', None)
        message = getattr(action, 'message', None)
        return message or None
    if name != "MessageEvent" or getattr(event, 'source', None) != "agent":
        return None
    if not isinstance(event, LLMConvertibleEvent):
        return None
    try:
        message = event.to_llm_message()
    except Exception:
        return None
    content = getattr(message, 'content', None)
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        parts = [getattr(c, 'text', None) for c in content]
        parts = [p for p in parts if p]
        return "\n".join(parts) if parts else None
    return None


def _metrics_dict(llm: Any) -> dict[str, Any]:
    """Extract accumulated token usage / cost from an LLM's metrics."""
    metrics = getattr(llm, 'metrics', None)
    if metrics is None:
        return {}
    usage = getattr(metrics, 'accumulated_token_usage', None)
    return {
        'accumulated_cost': getattr(metrics, 'accumulated_cost', None),
        'token_usage': _to_jsonable(usage) if usage is not None else None,
    }


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


# --- In-run solution validation --------------------------------------------
#
# CLEVER problem files are structured into sections delimited by marker lines
# of the exact form ``-- [SECTION_NAME]``. The task rules require the agent to
# edit only the designated sections and keep every tag, but weaker models
# drop/rename tags, rewrite fixed sections, or smuggle in axioms; these pure
# helpers back the harness-side corrective loop in :meth:`OpenHands.openhands`.

_SECTION_TAG_RE = re.compile(r'^-- \[([A-Z_]+)\][ \t]*\r?$', re.MULTILINE)
# Non-greedy, so ``/-- doc -/`` blocks are covered too. Nested block comments
# are not handled (irrelevant for a whitespace/comment-resilient comparison).
_LEAN_BLOCK_COMMENT_RE = re.compile(r'/-.*?-/', re.DOTALL)
_STATUS_FLAGS = ('OK', 'FAIL', 'ISSUE', 'MISMATCH')


def _split_sections(text: str) -> tuple[dict[str, str], list[str]]:
    """Split into ``{label: body}`` plus the ordered list of tag occurrences.

    A body spans from the end of its tag line to the next tag line (or EOF).
    A duplicated label keeps its first body but is still recorded in order.
    """
    matches = list(_SECTION_TAG_RE.finditer(text))
    sections: dict[str, str] = {}
    order: list[str] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        order.append(m.group(1))
        sections.setdefault(m.group(1), text[m.end():end])
    return sections, order


def _normalize_lean(s: str) -> str:
    """Collapse a Lean fragment to its comment-free, whitespace-free skeleton.

    Block comments (``/- ... -/``, doc comments included) go first, then
    ``--`` line comments (section-marker lines are kept whole, should one ever
    reach a body), then all whitespace — so comparisons built on this treat
    whitespace/comment-only edits as non-structural.
    """
    s = _LEAN_BLOCK_COMMENT_RE.sub('', s)
    lines = [line if _SECTION_TAG_RE.match(line) else line.split('--', 1)[0]
             for line in s.splitlines()]
    return ''.join('\n'.join(lines).split())


def _missing_sections(original: str, solution: str) -> list[str]:
    """Labels tagged in ``original`` whose tag is absent from ``solution``."""
    _, order = _split_sections(original)
    present, _ = _split_sections(solution)
    return [label for label in dict.fromkeys(order) if label not in present]


def _changed_fixed_sections(original: str, solution: str, rules: dict[str, str]) -> list[str]:
    """Fixed-rule labels whose body structurally differs (or is missing)."""
    orig_sections, order = _split_sections(original)
    sol_sections, _ = _split_sections(solution)
    return [label for label in dict.fromkeys(order)
            if rules.get(label) == 'fixed'
            and (label not in sol_sections
                 or _normalize_lean(sol_sections[label]) != _normalize_lean(orig_sections[label]))]


def _restore_fixed_sections(original: str, solution: str, labels: list[str]) -> str:
    """Splice each named section's original body back into ``solution``.

    Only the region between the section's tag line and the next tag line (or
    EOF) is replaced; everything else is preserved byte-for-byte. A label
    duplicated in the solution has its first occurrence restored (matching
    :func:`_split_sections`).
    """
    orig_sections, _ = _split_sections(original)
    targets = set(labels) & orig_sections.keys()
    matches = list(_SECTION_TAG_RE.finditer(solution))
    out: list[str] = []
    pos = 0
    for i, m in enumerate(matches):
        if m.group(1) in targets:
            targets.discard(m.group(1))
            end = matches[i + 1].start() if i + 1 < len(matches) else len(solution)
            out.append(solution[pos:m.end()])
            out.append(orig_sections[m.group(1)])
            pos = end
    out.append(solution[pos:])
    return ''.join(out)


def _added_axioms(original: str, solution: str) -> list[str]:
    """Top-level ``axiom`` declarations in ``solution`` absent from ``original``."""
    def decls(text: str) -> list[str]:
        return [line.strip() for line in text.splitlines() if line.startswith('axiom ')]
    known = set(decls(original))
    return [a for a in dict.fromkeys(decls(solution)) if a not in known]


_TOP_DECL_RE = re.compile(
    r"^(?:@\[[^\]]*\]\s*)?(?:noncomputable\s+|private\s+|protected\s+|partial\s+)*"
    r"(?:def|theorem|lemma|abbrev|structure|inductive)\s+([A-Za-z_][A-Za-z0-9_.']*)",
    re.MULTILINE)


def _declared_names(text: str) -> set[str]:
    """Top-level declaration names in a Lean fragment."""
    return set(_TOP_DECL_RE.findall(text))


def _redeclared_signatures(original: str, solution: str, rules: dict[str, str]) -> list[str]:
    """``SECTION:name`` pairs where an editable section re-declares a name
    already declared in the original file.

    A body section continues the fixed signature above it (e.g.
    ``[SPECIFICATION_BODY]`` after ``def generated_spec ... : Prop :=``);
    pasting a full ``def generated_spec ...`` into it duplicates the header
    when downstream stages reassemble the file from blocks, producing inputs
    that no longer elaborate. New helper names remain allowed.
    """
    known = _declared_names(original)
    sol_sections, _ = _split_sections(solution)
    return [f"{label}:{name}"
            for label, body in sol_sections.items()
            if rules.get(label) == 'editable'
            for name in sorted(_declared_names(body) & known)]


def _preamble(text: str) -> str:
    """Everything before the first section tag (imports etc.)."""
    m = _SECTION_TAG_RE.search(text)
    return text[:m.start()] if m else text


def _changed_preamble(original: str, solution: str) -> bool:
    """True when the pre-tag region structurally differs.

    Downstream evaluation reconstructs files from the labeled blocks only, so
    helper definitions smuggled into the preamble compile in-env but vanish at
    evaluation time ("Unknown identifier") — the preamble must stay untouched.
    """
    return _normalize_lean(_preamble(solution)) != _normalize_lean(_preamble(original))


def _restore_preamble(original: str, solution: str) -> str:
    """Splice the original pre-tag region back into ``solution``."""
    m = _SECTION_TAG_RE.search(solution)
    return _preamble(original) + (solution[m.start():] if m else '')


class OpenHands(BaseAgentic):
    """Wrapper for the OpenHands SDK theorem prover (OpenAI-compatible endpoint)."""
    PROVER_NAME = "openhands"
    CONFIG_CLASS = OpenHandsConfig

    def _read_key_file(self) -> tuple[Optional[str], str]:
        """Parse the key file (two-line: base URL + key, or single-line: key)."""
        lines = [
            line.strip()
            for line in (var.KEYS_DIRECTORY / self.config.api_key_file).read_text().splitlines()
            if line.strip()
        ]
        if not lines:
            raise ValueError(f"Empty key file: {self.config.api_key_file}")
        if len(lines) >= 2:
            return lines[0], lines[1]
        return None, lines[0]

    def _make_llm(self) -> LLM:
        """Build the litellm-backed LLM pointed at the configured endpoint."""
        base_url, api_key = self._read_key_file()
        kwargs: dict[str, Any] = {
            'model': self.config.model,
            'base_url': base_url or self.config.base_url,
            'api_key': SecretStr(api_key),
            'usage_id': self._get_name(),
            'timeout': self.config.http_timeout,
            'stream': self.config.stream,
        }
        if self.config.reasoning_effort is not None:
            kwargs['reasoning_effort'] = self.config.reasoning_effort
        if self.config.temperature is not None:
            kwargs['temperature'] = self.config.temperature
        if self.config.seed is not None:
            kwargs['seed'] = self.config.seed
        if self.config.max_output_tokens is not None:
            kwargs['max_output_tokens'] = self.config.max_output_tokens
        if self.config.max_input_tokens is not None:
            kwargs['max_input_tokens'] = self.config.max_input_tokens
        # extra_body is forwarded verbatim to the endpoint (not stripped by
        # drop_params). Merge the optional server hints; a server that doesn't
        # know a key simply ignores it.
        extra_body: dict[str, Any] = {}
        if self.config.report_cached_tokens:
            extra_body['return_cached_tokens_details'] = True
        if self.config.disable_thinking:
            extra_body['chat_template_kwargs'] = {'enable_thinking': False}
        if extra_body:
            kwargs['litellm_extra_body'] = extra_body
        return LLM(**kwargs)

    class _StreamingLLM(LLM):
        """LLM whose completions stream even without a caller token callback.

        The endpoint's proxy injects ``stream_options={'include_usage': True}`` into
        every upstream request, and vLLM rejects any NON-streaming request
        carrying it ("Stream options can only be defined when stream=True") —
        so every non-streaming call through the proxy 400s. The base
        ``completion()`` silently degrades ``stream=True`` to non-streaming
        when no ``on_token`` is wired (SDK #4014 fallback); supply a no-op
        sink instead so the request stays genuinely streaming.
        """

        def completion(self, messages, *args, **kwargs):  # type: ignore[override]
            if self.stream and kwargs.get('on_token') is None:
                kwargs['on_token'] = lambda *_a, **_k: None
            return super().completion(messages, *args, **kwargs)

    class _StreamingSummarizingCondenser(LLMSummarizingCondenser):
        """Condenser whose summary calls survive the endpoint's proxy.

        ``LLMSummarizingCondenser._disable_streaming_for_summary`` forces
        ``stream=False`` on its LLM (summaries have no token sink), which
        makes the first condensation 400 on the endpoint and kills the
        conversation (observed as ``ConversationRunError: Summarization LLM
        call failed ... Stream options can only be defined when stream=True``
        on 13/273 SPEC_ISO records, 2026-08-06). Keep the LLM streaming; the
        paired ``_StreamingLLM`` drains the tokens.
        """

        @model_validator(mode="after")
        def _disable_streaming_for_summary(self):  # override: keep stream=True
            return self

    def _make_condenser(self, llm: LLM) -> Optional[LLMSummarizingCondenser]:
        """Bound conversation history so long proof runs don't grow unbounded.

        Mirrors the SDK's default preset: an ``LLMSummarizingCondenser`` on a
        ``usage_id='condenser'`` copy of the agent LLM (separate cost/token
        attribution). ``max_tokens`` is the trigger; the event-count cap
        (``max_size``) is disabled by default so condensation only happens when
        the context window is actually full. Returns ``None`` when disabled
        (ablation).
        """
        if not self.config.condenser_enabled:
            return None
        max_tokens = self.config.condenser_max_tokens
        if max_tokens is None:
            if self.config.max_input_tokens:
                max_tokens = int(self.config.condenser_context_fraction * self.config.max_input_tokens)
            else:
                max_tokens = 100000  # conservative fallback when the window is unknown
        return self._StreamingSummarizingCondenser(
            llm=self._StreamingLLM(**{**dict(llm), 'usage_id': 'condenser'}),
            max_tokens=max_tokens,
            max_size=self.config.condenser_max_size or OPENHANDS_NO_EVENT_CAP,
            keep_first=self.config.condenser_keep_first,
        )

    def _make_mcp_config(self, paths: dict[str, Any]) -> dict[str, MCPServer]:
        """Build the stdio MCP server registry for the agent.

        ``cwd`` is pinned to the per-attempt env so the server (lean-lsp-mcp)
        resolves relative file paths from the prompt against the workspace
        rather than the harness process directory (parity with the Claude and
        Codex wrappers, which run the whole agent with cwd = the env).
        """
        return {
            key: MCPServer(transport=mcp.type, command=mcp.command, args=mcp.args or [],
                           cwd=str(paths['env']))
            for key, mcp in self.config.mcp_server.items()
        }

    def _skill_root(self, paths: dict[str, Any], skill: OpenHandsSkillConfig) -> Path:
        """On-disk root of a skill's plugin directory (e.g. .../plugins/lean4)."""
        return Path(paths['skills']) / skill.root / skill.subdir

    def _load_skill_docs(self, paths: dict[str, Any]) -> list[str]:
        """Inline each skill's SKILL.md into the system prompt.

        Used only when ``native_skills`` is False (``--no-native``): the legacy
        inline regime of the pre-2026-08 recorded runs. The default native
        regime instead loads the plugin through the SDK's Claude-Code-layout
        plugin system — see the ``native_skills`` config comment. The skill
        entry doc (SKILL.md) is injected verbatim, pointing at the plugin
        root where commands/ and skills/*/references/ can be read on demand.
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
                    f"terminal tool as needed. Skill entry follows:\n\n{content}")
        return docs

    def _make_agent_context(self, base_instructions: Optional[str], paths: dict[str, Any]) -> Optional[AgentContext]:
        """Append the task instructions and skill content to the system prompt.

        Mirrors the Claude wrapper's "append to the built-in preset" behaviour:
        OpenHands keeps its own coding system prompt and our task spec + the
        lean4 skill are layered on via ``system_message_suffix``.
        """
        suffix_parts: list[str] = []
        if base_instructions:
            suffix_parts.append(base_instructions)
        if not self.config.native_skills:
            docs = self._load_skill_docs(paths)
            if docs:
                suffix_parts.extend(docs)
            elif self.config.skills:
                # No SKILL.md found (non-standard skill): fall back to advertising paths.
                lines = [f"- {key}: {skill.description} (files under {self._skill_root(paths, skill)})"
                         for key, skill in self.config.skills.items()]
                suffix_parts.append("Available skills — read their files with the terminal tool:\n"
                                    + "\n".join(lines))
        if not suffix_parts:
            return None
        # current_datetime=None drops the injected "<CURRENT_DATETIME>" section so
        # the system prefix is byte-identical across attempts — lets a server-side
        # prefix cache hit across problems, not just within one conversation.
        return AgentContext(system_message_suffix="\n\n".join(suffix_parts),
                            current_datetime=None)

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
        which are opaque strings in the inline-skills regime. A prompt that *is*
        a slash command is rewritten into the command's markdown definition; a
        prompt that merely *references* commands mid-text (e.g. the END_TO_END
        task) gets each referenced command's definition appended. Non-command
        prompts pass through unchanged, and with ``--no-skills`` (empty
        ``skills``) no command file is found, so the raw slash command is left
        as-is (clean ablation). Skipped entirely when ``native_skills`` is on
        (the SDK plugin loader registers the commands itself).
        """
        m = _SLASH_CMD_RE.match(prompt.strip())
        if m is not None:
            plugin, cmd, rest = m.group('plugin'), m.group('cmd'), m.group('rest').strip()
            cmd_file = self._find_command_file(paths, plugin, cmd)
            if cmd_file is None:
                return prompt
            body = _strip_frontmatter(cmd_file.read_text()).strip()
            root = cmd_file.parent.parent  # plugin root (commands/<cmd>.md -> ../..)
            return (
                f"Execute the `/{plugin}:{cmd}` command with arguments: `{rest}`.\n"
                f"Its definition is below. Paths it references (e.g. `../skills/...`, "
                f"`references/...`) are relative to the command file `{cmd_file}` "
                f"(plugin root `{root}`) — read them with the terminal tool as needed.\n\n"
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
            "relative to its command file — read them with the terminal tool as "
            "needed.\n\n" + "\n\n".join(blocks))

    def _skill_env(self, paths: dict[str, Any]) -> dict[str, str]:
        """Env vars each plugin's SessionStart bootstrap hook would export.

        Claude Code runs ``hooks/bootstrap.sh`` at session start, which exports
        e.g. ``LEAN4_SCRIPTS`` — the inlined SKILL.md invokes its scripted
        primitives through these variables. Outside Claude Code the hook never
        runs, so execute it here (with ``CLAUDE_PLUGIN_ROOT``/``CLAUDE_ENV_FILE``
        set the way Claude Code would) and collect the exported variables. The
        result is applied to the worker's ``os.environ``, which both the
        terminal tool and the stdio MCP subprocesses inherit.
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

    def _plugin_sources(self, paths: dict[str, Any]) -> list[PluginSource]:
        """Native plugin sources for ``Conversation(plugins=...)``.

        The SDK reads the Claude-Code plugin layout itself
        (.claude-plugin/plugin.json, skills/, commands/): skills are listed
        with progressive disclosure and each ``commands/<cmd>.md`` becomes a
        skill keyword-triggered by the literal ``/<plugin>:<cmd>`` string.
        The plugin's Claude-style hooks are inert here (matcher 'Bash' never
        matches the 'terminal' tool and ``${CLAUDE_PLUGIN_ROOT}`` is not
        substituted) — their env exports are provided by :meth:`_skill_env`
        instead.
        """
        return [PluginSource(source=str(self._skill_root(paths, skill)))
                for skill in self.config.skills.values()]

    def _make_workspace(self, paths: dict[str, Any]) -> Any:
        """Return the conversation workspace.

        Local (in-process) runtime: the per-attempt temp env directory. Swap for
        a ``DockerWorkspace`` here to gain OpenHands' containerised sandbox.
        """
        return str(paths['env'])

    def openhands(self, problem: Problem, prompt: str, paths: dict[str, Path]) -> tuple[
        list[dict[str, Any]], Optional[str]]:
        """Run the OpenHands agent to completion and collect the trace."""
        _quiet_dependencies()
        overrides = {**self.config.dns_overrides, **_parse_host_overrides(os.environ.get(_HOST_RESOLVE_ENV))}
        _install_host_overrides(overrides)
        # Terminal tool and stdio MCP subprocesses inherit the worker process
        # env — apply the plugin bootstrap-hook exports (LEAN4_SCRIPTS etc.).
        os.environ.update(self._skill_env(paths))
        if self.config.context is not None:
            src = var.CONTEXTS_DIRECTORY / self.config.context.src
            base_instructions = src.read_text()
            if self.config.context.dst is not None:
                (paths['env'] / self.config.context.dst).write_text(base_instructions)
        else:
            base_instructions = None

        llm = self._make_llm()
        # Detach the agent's bash session from the user's shell history:
        # HISTFILE=/dev/null keeps agent commands out of ~/.bash_history /
        # ~/.zsh_history AND starts the session with no pre-loaded history
        # (the terminal is an interactive /bin/bash inside tmux, which would
        # otherwise read and append the inherited history file). Note this is
        # hygiene, not containment — the tool is unsandboxed (see module
        # docstring), so the model could still `cat` a history file on disk.
        terminal_env = {'HISTFILE': '/dev/null', 'HISTSIZE': '0',
                        'HISTFILESIZE': '0', 'SAVEHIST': '0'}
        agent_kwargs: dict[str, Any] = dict(
            llm=llm,
            tools=[Tool(name=TerminalTool.name, params={'env': terminal_env}),
                   Tool(name=FileEditorTool.name)],
            mcp_config=self._make_mcp_config(paths),
            agent_context=self._make_agent_context(base_instructions, paths),
            condenser=self._make_condenser(llm),
        )
        if self.config.tool_filter_regex is not None:
            agent_kwargs['filter_tools_regex'] = self.config.tool_filter_regex
        agent = Agent(**agent_kwargs)

        trace: list[dict[str, Any]] = list()
        status: Optional[str] = None
        events: list[Any] = list()
        conv_ref: list[Any] = list()
        problem.reset_timer()

        def on_event(event: Event) -> None:
            nonlocal status
            events.append(event)
            trace.append(_serialize_event(event))
            logging.debug(f"New OpenHands event: {trace[-1]}")
            if status != "TIMEOUT" and problem.get_elapsed() > self.config.generation_timeout:
                if conv_ref:
                    try:
                        conv_ref[0].pause()
                    except Exception as exc:  # pragma: no cover - best-effort pause
                        logging.debug(f"OpenHands pause failed: {exc}")
                logging.warning(
                    f"OpenHands generation timed out after {self.config.generation_timeout} seconds.")
                status = "TIMEOUT"

        # A no-op token sink: the SDK degrades a stream=True request back to
        # non-streaming unless a token callback is wired (llm.py "Streaming
        # requested without an on_token callback"), which would drop the
        # stream=True the endpoint proxy's injected stream_options needs. We consume
        # the trace from Events (on_event), not tokens, so the deltas are
        # discarded.
        def on_token(_chunk: Any) -> None:
            return None

        conv_kwargs: dict[str, Any] = dict(
            agent=agent,
            workspace=self._make_workspace(paths),
            callbacks=[on_event],
            token_callbacks=[on_token] if self.config.stream else None,
            max_iteration_per_run=self.config.max_turns or OPENHANDS_NO_ITERATION_LIMIT,
            # The SDK's stuck detector (on by default) silently ends run() after
            # a few identical tool-call/observation pairs — a pattern Lean proof
            # loops hit legitimately — with no exception and no trace event.
            # Disable it for parity with the Claude wrapper (which has none).
            stuck_detection=False,
        )
        if self.config.native_skills:
            conv_kwargs['plugins'] = self._plugin_sources(paths)
        # Mute the SDK's rich per-event console visualizer unless we are debugging
        # (multiple workers streaming to one terminal is unreadable). Full trace
        # is always preserved in problem.metadata regardless.
        if not logging.getLogger().isEnabledFor(logging.DEBUG):
            conv_kwargs['visualizer'] = None
        conversation = Conversation(**conv_kwargs)
        conv_ref.append(conversation)

        def run_message(text: str) -> Optional[str]:
            """Send a message, run to completion, return the last assistant text."""
            before = len(events)
            conversation.send_message(text)
            try:
                conversation.run()
            except Exception as exc:
                # run() wraps everything in ConversationRunError; the chain walk
                # inside converts wrapped rate limits into a retry request.
                _raise_for_rate_limit(exc)
                raise
            reply = None
            for event in events[before:]:
                text_out = _assistant_text(event)
                if text_out is not None:
                    reply = text_out
            return reply

        # Snapshot the input problem files before the main turn: the in-run
        # validation below compares the agent's edits against these. Files the
        # agent creates later (e.g. solutions/*.lean) are not checked.
        rules = self.config.validation_sections or {}
        originals = {p: p.read_text() for p in paths['problems'] if p.is_file()} if rules else {}

        def run_validation() -> bool:
            """Corrective loop: dropped tags (Check A), rewritten fixed sections (Check B).

            Each check has its own ``validation_max_retries`` round budget
            (shared across files). Corrective turns go through the same
            conversation as the status turn and are bounded the same way:
            ``on_event`` pauses the run and flips ``status`` to TIMEOUT at the
            wall clock. Returns False when a budget runs out with violations
            remaining.
            """
            rounds = {'tags': 0, 'fixed': 0}
            for path, original in originals.items():
                while status != "TIMEOUT":
                    solution = path.read_text()
                    # Check A first: Check B needs the tags in place to compare.
                    missing = _missing_sections(original, solution)
                    changed = [] if missing else _changed_fixed_sections(original, solution, rules)
                    axioms = [] if missing else _added_axioms(original, solution)
                    pre = False if missing else _changed_preamble(original, solution)
                    redecl = [] if missing else _redeclared_signatures(original, solution, rules)
                    if not (missing or changed or axioms or pre or redecl):
                        break
                    check = 'tags' if missing else 'fixed'
                    if rounds[check] >= self.config.validation_max_retries:
                        return False
                    rounds[check] += 1
                    if missing:
                        restored, violations = False, missing
                        message = OPENHANDS_VALIDATION_PROMPT['tags'].format(
                            missing=missing, filename=path.name)
                    else:
                        restored = bool(changed or pre)
                        violations = (changed + (['__PREAMBLE__'] if pre else [])
                                      + (['AXIOM_ADDED'] if axioms else [])
                                      + [f'REDECL:{r}' for r in redecl])
                        parts = []
                        new_text = solution
                        if changed:
                            new_text = _restore_fixed_sections(original, new_text, changed)
                            parts.append(OPENHANDS_VALIDATION_PROMPT['fixed'].format(
                                changed=changed, filename=path.name))
                        if pre:
                            new_text = _restore_preamble(original, new_text)
                            parts.append(OPENHANDS_VALIDATION_PROMPT['preamble'].format(
                                filename=path.name))
                        if restored:
                            # Harness-side restore; _solve re-reads the file from
                            # disk afterwards, so the write lands in the record.
                            path.write_text(new_text)
                        if axioms:
                            parts.append(OPENHANDS_VALIDATION_PROMPT['axiom'].format(axioms=axioms))
                        if redecl:
                            parts.append(OPENHANDS_VALIDATION_PROMPT['redecl'].format(redecl=redecl))
                        message = " ".join(parts)
                    trace.append({'type': 'ValidationMessage', 'check': check, 'round': rounds[check],
                                  'violations': violations, 'restored': restored})
                    run_message(message)
            return True

        try:
            main_reply = run_message(prompt)
            trace.append({'type': 'ResultMessage',
                          'result': main_reply if main_reply is not None else "",
                          'model': self.config.model,
                          'usage': _metrics_dict(llm),
                          'native_skills': self.config.native_skills,
                          'total_cost_usd': getattr(getattr(llm, 'metrics', None), 'accumulated_cost', None)})

            validation_ok = True
            if originals and status != "TIMEOUT":
                validation_ok = run_validation()
                if not validation_ok and status != "TIMEOUT":
                    # Budget exhausted with violations left: fail outright and
                    # skip the status turn (its flag would not be trustworthy).
                    status = 'FAIL'
                    trace.append({'type': 'StatusMessage', 'status': status,
                                  'usage': _metrics_dict(llm)})

            if self.config.status_prompt is not None and status != "TIMEOUT" and validation_ok:
                status_text = OPENHANDS_STATUS_PROMPT[self.config.status_prompt]
                status_reply = run_message(status_text)
                if status is None and status_reply is not None:
                    # Strip markdown emphasis some models wrap the flag in
                    # ("**ISSUE**" -> "ISSUE") so consumers match exact strings.
                    status = status_reply.strip().strip('*_#').strip()
                # Check C (validation runs only): insist on a bare flag word.
                # The -mult prompt expects multi-line "<i>:<flag>" replies, so
                # it is exempt.
                if rules and self.config.status_prompt != 'mismatch-spec-mult':
                    for n in range(1, self.config.validation_max_retries + 1):
                        if status == "TIMEOUT" or status in _STATUS_FLAGS:
                            break
                        trace.append({'type': 'ValidationMessage', 'check': 'flag', 'round': n,
                                      'violations': [status], 'restored': False})
                        reply = run_message(status_text + OPENHANDS_VALIDATION_PROMPT['flag'])
                        if reply is not None and status != "TIMEOUT":
                            status = reply.strip().strip('*_#').strip()
                    if status != "TIMEOUT" and status not in _STATUS_FLAGS:
                        status = 'FAIL'
                trace.append({'type': 'StatusMessage', 'status': status, 'usage': _metrics_dict(llm)})

            # The status turn runs with full tool access, so the model can
            # re-tamper the file AFTER Checks A/B passed (observed once:
            # a theorem-statement edit during the flag turn). Re-check once,
            # silently: splice-restore what is restorable (fixed sections,
            # preamble) without another model turn and record the event; the
            # flag is kept — the record stays structurally sound and the
            # evaluator judges the rest.
            if originals:
                for path, original in originals.items():
                    solution = path.read_text()
                    post_missing = _missing_sections(original, solution)
                    post_changed = [] if post_missing else _changed_fixed_sections(original, solution, rules)
                    post_axioms = [] if post_missing else _added_axioms(original, solution)
                    post_pre = False if post_missing else _changed_preamble(original, solution)
                    post_redecl = [] if post_missing else _redeclared_signatures(original, solution, rules)
                    if not (post_missing or post_changed or post_axioms or post_pre or post_redecl):
                        continue
                    new_text = solution
                    if post_changed:
                        new_text = _restore_fixed_sections(original, new_text, post_changed)
                    if post_pre:
                        new_text = _restore_preamble(original, new_text)
                    if new_text != solution:
                        path.write_text(new_text)
                    trace.append({'type': 'ValidationMessage', 'check': 'post', 'round': 1,
                                  'violations': (post_missing + post_changed
                                                 + (['__PREAMBLE__'] if post_pre else [])
                                                 + (['AXIOM_ADDED'] if post_axioms else [])
                                                 + [f'REDECL:{r}' for r in post_redecl]),
                                  'restored': new_text != solution})
        finally:
            try:
                conversation.close()
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logging.debug(f"OpenHands conversation close failed: {exc}")

        return trace, status

    def _agent(self, problem: Problem, paths: dict[str, Path | list[Path]] | None = None, **kwargs: Any) -> None:
        """Build the prompt and invoke the OpenHands agent."""
        prompt = self.config.prompt.format(filepath=" ".join(s.name for s in paths['problems']))
        if not self.config.native_skills:
            # Native plugins register each command as a '/plugin:cmd'-triggered
            # skill, so the raw slash command is resolved by the SDK itself.
            prompt = self._expand_prompt(prompt, paths)
        try:
            trace, status = self.openhands(problem=problem, prompt=prompt, paths=paths)
        except RetryGenerationException:
            raise
        except Exception as exc:
            _raise_for_rate_limit(exc)  # walks the wrapped-exception chain
            # Transient endpoint blip (Open WebUI 5xx, connection reset, ...):
            # re-run the whole attempt in a fresh env, up to a bounded count.
            if _is_transient_error(exc):
                n = problem.metadata.get('_transient_retries', 0) + 1
                problem.metadata['_transient_retries'] = n
                if n <= self.config.max_transient_retries:
                    logging.warning(
                        f"OpenHands transient endpoint error (re-run {n}/"
                        f"{self.config.max_transient_retries}) for {problem.info.id}: {exc}")
                    raise RetryGenerationException(
                        message=str(exc), wait_seconds=self.config.transient_retry_delay)
                logging.error(
                    f"OpenHands transient error exceeded {self.config.max_transient_retries} "
                    f"re-runs for {problem.info.id}; recording failure: {exc}")
            raise
        # Keyed on the bare prover name (like claude.py's literal 'claude'):
        # consumers (clever.py summaries, display.py) look the trace up under
        # the -p prover id, which never carries the name_modifier suffix.
        problem.metadata.update({'prompt': prompt, self.PROVER_NAME: trace, 'status': status})
        try:
            result = next(t['result'] for t in trace if t.get('type', None) == "ResultMessage")
        except StopIteration:
            logging.info("OpenHands exited before returning a result message")
            raise
        except KeyError:
            logging.error("No summary found in result message!")
            raise
        problem.metadata['result'] = result


# --- Named model variants exposed as distinct -p backends -------------------
#
# Each variant fixes a model id served by the endpoint. ``PROVER_NAME`` is the -p label:
# outputs land in ``data/<label>`` (plus any name_modifier suffix) and the
# trace is keyed under the bare ``<label>``, keeping save / load / summary /
# display consistent.

### KIMI K2.5

@dataclass
class KimiConfig(OpenHandsConfig):
    model: str = "openai/Kimi"
    max_input_tokens: Optional[int] = 262144  # served 256k window (confirmed from endpoint)


class Kimi(OpenHands):
    """Moonshot Kimi served by the institutional endpoint via OpenHands."""
    PROVER_NAME = "kimi"
    CONFIG_CLASS = KimiConfig


### MINIMAX M2.7

@dataclass
class MiniMaxConfig(OpenHandsConfig):
    model: str = "openai/MiniMax"


class MiniMax(OpenHands):
    """MiniMax served by the institutional endpoint via OpenHands."""
    PROVER_NAME = "minimax"
    CONFIG_CLASS = MiniMaxConfig


### GLM 5.2

@dataclass
class GlmConfig(OpenHandsConfig):
    model: str = "openai/GLM-5.2-nvfp4"


class Glm(OpenHands):
    """GLM-5.2 (nvfp4) served by the institutional endpoint via OpenHands."""
    PROVER_NAME = "glm"
    CONFIG_CLASS = GlmConfig


### QWEN3 235B-A22B

@dataclass
class QwenConfig(OpenHandsConfig):
    # Qwen3-235B-A22B (FP8) on a LAN vLLM server (port 8800, reached via
    # localhost — run on the host or over an SSH tunnel) — not the
    # institutional proxy, hence its own base_url and key file. Served model id,
    # FP8 weights and the 131,072 max_model_len all confirmed against
    # `GET /v1/models` (2026-09-18).
    model: str = "openai/Qwen3-235B-A22B"
    base_url: Optional[str] = "http://localhost:8800/v1"
    api_key_file: str = "vllm"
    # 131,072-token window via YaRN rope scaling (server-side vLLM config; the
    # client only needs the budget so the condenser triggers at the right size).
    max_input_tokens: Optional[int] = 131072
    # Per-turn response cap (thinking + output; maps to max_completion_tokens
    # on each request). Raw vLLM is otherwise unbounded, unlike the hosted
    # Kimi endpoint: server counters recorded 472 requests ended by the
    # context-window limit, 4 generating >100k tokens with 16-32 min decodes.
    # 32768 amputates that runaway tail while binding on only 2 of 8,605
    # recorded SPEC_GEN turns (exact-tokenizer census, 2026-09-19); the
    # qualitative audit found no productive reasoning past ~24k tokens.
    max_output_tokens: Optional[int] = 32768


class Qwen(OpenHands):
    """Qwen3-235B-A22B (FP8, YaRN 128k) served by a LAN vLLM server via OpenHands.

    Everything else (skills regime, condenser policy, validation, streaming,
    timeouts) inherits the shared OpenHands defaults — same setup as the Kimi
    variant for cross-model comparability.
    """
    PROVER_NAME = "qwen"
    CONFIG_CLASS = QwenConfig
