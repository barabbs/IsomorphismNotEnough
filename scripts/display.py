"""Interactive TUI explorer for Problem objects."""

import argparse
import json
from pathlib import Path

from rich.columns import Columns
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt

from core.problem import Problem, ProblemInfo
from core.benchmarks.clever import TASKS
from core import var

console = Console()


def _header(problem: Problem) -> str:
    info = problem.info
    meta = problem.metadata
    status = meta.get("status", "N/A")
    comment = meta.get("comment", "")
    timing = f"{problem.timing:.2f}s" if problem.timing is not None else "N/A"
    exc = ""
    if problem.exception:
        exc = f"\n  Exception: {problem.exception['type']}: {problem.exception['value']}"
    comment_line = f"  {comment}" if comment else ""
    return (
        f"  {info.dataset}/{info.subset}/{info.get_str_id()}"
        f"  —  {info.task} (attempt {info.attempt})\n"
        f"  Status: {status}{comment_line}    Timing: {timing}{exc})"
    )


def _show_text(content: str | list[str] | None, title: str, lang: str = "lean4") -> None:
    """Display a text field. If it's a list, show an interactive entry browser."""
    if not content:
        console.print(f"  [dim](no {title})[/dim]")
        return

    if isinstance(content, str):
        console.print(Syntax(content, lang, theme="monokai", line_numbers=True, word_wrap=True), overflow="fold")
        return

    # Multi-entry list: interactive browser
    n = len(content)
    idx = 0
    while True:
        entry = content[idx]
        # Show a short preview of first line for context
        console.print(Panel.fit(
            f"[bold]{title}[/bold]  entry [cyan]{idx + 1}[/cyan] / {n}",
            border_style="green",
        ))
        if entry:
            console.print(Syntax(entry, lang, theme="monokai", line_numbers=True, word_wrap=True), overflow="fold")
        else:
            console.print("  [dim](empty)[/dim]")

        if n == 1:
            break

        nav = Prompt.ask(
            "[bold]n[/]=next, [bold]p[/]=prev, [bold]#[/]=jump, [bold]b[/]=back",
            default="n"
        ).strip().lower()
        if nav in ("b", "back", "q"):
            break
        elif nav in ("n", ""):
            idx = (idx + 1) % n
        elif nav == "p":
            idx = (idx - 1) % n
        else:
            try:
                jump = int(nav)
                if 1 <= jump <= n:
                    idx = jump - 1
                else:
                    console.print(f"[red]Out of range (1-{n})[/red]")
            except ValueError:
                console.print("[red]Invalid input[/red]")


def _msg_preview(msg: dict) -> tuple[str, str, str]:
    """Extract (content_kind, tokens_info, preview) from a message."""
    mtype = msg.get("type", "?")
    content = msg.get("content")
    usage = msg.get("usage", {})

    # Tokens column: show input/output for AssistantMessage, total for ResultMessage
    tokens = ""
    if usage and isinstance(usage, dict):
        out_t = usage.get("output_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_create = usage.get("cache_creation_input_tokens", 0)
        parts = []
        if out_t:
            parts.append(f"out:{out_t}")
        if cache_read:
            parts.append(f"cache:{cache_read}")
        if cache_create:
            parts.append(f"new:{cache_create}")
        tokens = " ".join(parts)

    # Content kind + preview
    kind = ""
    preview = ""

    if mtype == "AssistantMessage" and isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if "name" in item:
                # Tool use
                kind = f"tool:{item['name']}"
                inp = item.get("input", {})
                if isinstance(inp, dict):
                    # Show the most informative input arg
                    for key in ("file_path", "command", "query", "pattern", "content"):
                        if key in inp:
                            val = str(inp[key])
                            preview = f"{key}={val}"
                            break
                    else:
                        preview = ", ".join(f"{k}=" for k in list(inp.keys())[:3])
            elif "text" in item:
                kind = "text"
                preview = item["text"].strip()
            elif "thinking" in item:
                kind = "thinking"
                preview = item["thinking"].strip()

    elif mtype == "UserMessage":
        kind = "tool_result"
        tr = msg.get("tool_use_result", "")
        if isinstance(tr, str):
            preview = tr
        elif isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict):
                is_err = first.get("is_error", False)
                kind = "tool_error" if is_err else "tool_result"
                c = first.get("content", "")
                preview = str(c)

    elif mtype == "SystemMessage":
        sub = msg.get("subtype", "")
        kind = sub
        data = msg.get("data", {})
        if isinstance(data, dict):
            preview = data.get("hook_name", data.get("type", ""))

    elif mtype == "RateLimitEvent":
        info = msg.get("rate_limit_info", {})
        kind = info.get("status", "")
        util = info.get("utilization")
        if util is not None:
            preview = f"utilization: {util}"

    elif mtype == "ResultMessage":
        sub = msg.get("subtype", "")
        kind = sub
        cost = msg.get("total_cost_usd")
        dur = msg.get("duration_ms")
        turns = msg.get("num_turns")
        parts = []
        if cost is not None:
            parts.append(f"${cost:.4f}")
        if dur is not None:
            parts.append(f"{dur / 1000:.1f}s")
        if turns is not None:
            parts.append(f"{turns} turns")
        tokens = msg.get("stop_reason", "")
        preview = "  ".join(parts)

    # Truncate preview
    preview = preview.replace("\n", " ").strip()
    if len(preview) > 90:
        preview = preview[:90] + "…"

    return kind, tokens, preview


def _parse_msg_selection(text: str, total: int) -> list[int] | None:
    """Parse a message selection string into a list of indices.

    Supports single numbers (5), ranges (3-7), and comma-separated combinations (1,3-5,8).
    Returns None for quit/back, empty list for invalid input.
    """
    text = text.strip()
    if text.lower() in ("b", "back", "q"):
        return None
    indices = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            try:
                start, end = part.split("-", 1)
                start, end = int(start), int(end)
                indices.extend(range(max(0, start), min(total, end + 1)))
            except ValueError:
                console.print(f"[red]Invalid range: {part}[/red]")
                return []
        else:
            try:
                idx = int(part)
                if 0 <= idx < total:
                    indices.append(idx)
                else:
                    console.print(f"[red]Out of range: {idx}[/red]")
                    return []
            except ValueError:
                console.print(f"[red]Invalid selection: {part}[/red]")
                return []
    return indices


def _render_message(msg: dict, idx: int) -> None:
    """Render a single Claude conversation message in a human-friendly format."""
    mtype = msg.get("type", "?")
    subtype = msg.get("subtype", "")
    content = msg.get("content")

    type_label = {
        "AssistantMessage": "[green bold]Assistant[/]",
        "UserMessage": "[cyan bold]User[/]",
        "SystemMessage": "[yellow bold]System[/]",
        "RateLimitEvent": "[dim]RateLimit[/]",
        "ResultMessage": "[magenta bold]Result[/]",
    }.get(mtype, mtype)
    if subtype:
        type_label += f" [dim]/ {subtype}[/]"

    # Model tag for assistant messages
    model = msg.get("model", "")
    header = f"#{idx}  {type_label}"
    if model:
        header += f"  [dim]({model})[/dim]"

    console.print()
    console.rule(header)

    if mtype == "AssistantMessage" and isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if "thinking" in item:
                thinking = item["thinking"]
                console.print(Panel(
                    thinking,
                    title="[dim italic]thinking[/]",
                    border_style="dim",
                    padding=(0, 1),
                ))
            elif "text" in item:
                console.print(Markdown(item["text"]))
            elif "name" in item:
                # Tool use
                name = item["name"]
                inp = item.get("input", {})
                console.print(f"  [bold yellow]{name}[/]")
                if isinstance(inp, dict):
                    for k, v in inp.items():
                        v_str = str(v)
                        if len(v_str) > 500:
                            v_str = v_str[:500] + "…"
                        if "\n" in v_str or len(v_str) > 100:
                            # Multi-line value: show as indented block
                            console.print(f"    [dim]{k}:[/]")
                            # Guess language for syntax highlighting
                            lang = "lean4" if k == "content" and ("import" in v_str or "theorem" in v_str) else "text"
                            console.print(Syntax(v_str, lang, theme="monokai", padding=1, word_wrap=True))
                        else:
                            console.print(f"    [dim]{k}:[/] {v_str}")

    elif mtype == "UserMessage":
        is_err = False
        result_text = ""
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict):
                is_err = first.get("is_error", False)
                result_text = str(first.get("content", ""))
        if not result_text:
            result_text = str(msg.get("tool_use_result", ""))

        if is_err:
            console.print(Panel(result_text, title="[red bold]Error[/]", border_style="red", padding=(0, 1)))
        elif result_text:
            if len(result_text) > 2000:
                console.print(f"  [dim]({len(result_text)} chars — showing first 2000)[/dim]")
                result_text = result_text[:2000] + "…"
            console.print(Syntax(result_text, "text", theme="monokai", word_wrap=True, padding=1))

    elif mtype == "SystemMessage":
        data = msg.get("data", {})
        if isinstance(data, dict) and data:
            hook = data.get("hook_name", "")
            if hook:
                console.print(f"  hook: [bold]{hook}[/]")
            # Show other interesting fields
            for k in ("hook_event", "session_id"):
                if k in data:
                    console.print(f"  [dim]{k}: {data[k]}[/]")

    elif mtype == "RateLimitEvent":
        info = msg.get("rate_limit_info", {})
        status = info.get("status", "?")
        style = "green" if status == "allowed" else "red"
        console.print(f"  Status: [{style}]{status}[/]")
        for k in ("rate_limit_type", "utilization"):
            if info.get(k) is not None:
                console.print(f"  [dim]{k}: {info[k]}[/]")

    elif mtype == "ResultMessage":
        pairs = [
            ("Status", f"[green]{subtype}[/]" if subtype == "success" else f"[red]{subtype}[/]"),
            ("Cost", f"${msg.get('total_cost_usd', 0):.4f}"),
            ("Duration", f"{msg.get('duration_ms', 0) / 1000:.1f}s"),
            ("API time", f"{msg.get('duration_api_ms', 0) / 1000:.1f}s"),
            ("Turns", str(msg.get("num_turns", "?"))),
            ("Stop reason", msg.get("stop_reason", "?")),
        ]
        usage = msg.get("usage", {})
        if usage:
            pairs.append(("Output tokens", str(usage.get("output_tokens", "?"))))
            pairs.append(("Cache read", str(usage.get("cache_read_input_tokens", "?"))))
            pairs.append(("Cache created", str(usage.get("cache_creation_input_tokens", "?"))))

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="dim")
        table.add_column()
        for label, val in pairs:
            table.add_row(label, val)
        console.print(table)

    console.print()


def _openai_event_preview(msg: dict) -> tuple[str, str, str]:
    """Extract (short_type, kind, preview) from an OpenAI stream event."""
    etype = msg.get("type", "?")
    name = msg.get("name", "")
    short = etype.replace("_stream_event", "").replace("_event", "")
    if name:
        short = name
    kind, preview = "", ""

    item = msg.get("item") or {}
    raw = item.get("raw_item") or {}
    data = msg.get("data") or {}

    if etype == "run_item_stream_event":
        itype = item.get("type", "")
        if name == "tool_called":
            tool_name = raw.get("name") or raw.get("type") or item.get("title") or "tool"
            kind = f"tool:{tool_name}"
            # Try to surface the most informative argument.
            for key in ("commands", "command", "action", "arguments", "input", "query"):
                if key in raw and raw[key]:
                    val = raw[key]
                    if key == "action" and isinstance(val, dict):
                        val = val.get("commands", val)
                    preview = f"{key}={val}"
                    break
            else:
                preview = ", ".join(f"{k}" for k in list(raw.keys())[:4])
        elif name == "tool_output":
            kind = "tool_output"
            out = msg.get("item", {}).get("output")
            preview = str(out) if out is not None else str(raw.get("output", ""))
        elif name == "message_output_created":
            kind = "text"
            # raw_item is a ResponseOutputMessage with content list
            content = raw.get("content") or []
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                    preview = part.get("text", "")
                    break
        elif name == "reasoning_item_created":
            kind = "reasoning"
            summary = raw.get("summary")
            if isinstance(summary, list) and summary:
                first = summary[0]
                preview = first.get("text", "") if isinstance(first, dict) else str(first)
            else:
                preview = raw.get("id", "")
        else:
            kind = itype
    elif etype == "raw_response_event":
        dtype = data.get("type", "")
        kind = dtype
        if dtype == "response.output_text.done":
            preview = data.get("text", "")
        elif dtype == "response.shell_call_command.added":
            preview = data.get("command", "")
        elif dtype == "response.completed":
            resp = data.get("response") or {}
            usage = resp.get("usage") or {}
            in_t = usage.get("input_tokens", 0)
            out_t = usage.get("output_tokens", 0)
            preview = f"in:{in_t} out:{out_t}"
    elif etype == "agent_updated_stream_event":
        kind = "agent_updated"
        preview = msg.get("new_agent", "")
    elif etype == "ResultMessage":
        kind = "result"
        preview = str(msg.get("result", ""))
    elif etype == "StatusMessage":
        kind = "status"
        preview = str(msg.get("status", ""))

    preview = str(preview).replace("\n", " ").strip()
    if len(preview) > 90:
        preview = preview[:90] + "…"
    return short, kind, preview


def _render_openai_event(msg: dict, idx: int) -> None:
    """Render a single OpenAI stream event in a human-friendly format."""
    etype = msg.get("type", "?")
    name = msg.get("name", "")
    item = msg.get("item") or {}
    raw = item.get("raw_item") or {}
    data = msg.get("data") or {}

    type_label = {
        "run_item_stream_event": "[green bold]RunItem[/]",
        "raw_response_event": "[dim]Raw[/]",
        "agent_updated_stream_event": "[yellow]AgentUpdate[/]",
        "ResultMessage": "[magenta bold]Result[/]",
        "StatusMessage": "[magenta]Status[/]",
    }.get(etype, etype)
    if name:
        type_label += f" [dim]/ {name}[/]"

    console.print()
    console.rule(f"#{idx}  {type_label}")

    if etype == "run_item_stream_event":
        if name == "tool_called":
            tool_name = raw.get("name") or raw.get("type") or item.get("title") or "tool"
            console.print(f"  [bold yellow]{tool_name}[/]")
            # action.commands shape (ShellTool) or arguments str (function tools)
            action = raw.get("action") or {}
            commands = action.get("commands") if isinstance(action, dict) else None
            if commands:
                for cmd in commands:
                    console.print(Syntax(str(cmd), "bash", theme="monokai", word_wrap=True, padding=(0, 1)))
            else:
                for k in ("arguments", "input", "query", "command", "commands"):
                    if k in raw and raw[k] not in (None, ""):
                        v = raw[k]
                        v_str = json.dumps(v, indent=2, default=str) if not isinstance(v, str) else v
                        if len(v_str) > 2000:
                            v_str = v_str[:2000] + "…"
                        console.print(f"    [dim]{k}:[/]")
                        console.print(Syntax(v_str, "json" if not isinstance(v, str) else "text",
                                             theme="monokai", word_wrap=True, padding=(0, 1)))
        elif name == "tool_output":
            out = item.get("output")
            if out is None:
                out = raw.get("output", "")
            out_str = out if isinstance(out, str) else json.dumps(out, indent=2, default=str)
            if len(out_str) > 4000:
                console.print(f"  [dim]({len(out_str)} chars — showing first 4000)[/dim]")
                out_str = out_str[:4000] + "…"
            console.print(Syntax(out_str, "text", theme="monokai", word_wrap=True, padding=(0, 1)))
        elif name == "message_output_created":
            text_parts = []
            for part in raw.get("content") or []:
                if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                    text_parts.append(part.get("text", ""))
            if text_parts:
                console.print(Markdown("\n\n".join(text_parts)))
        elif name == "reasoning_item_created":
            summary = raw.get("summary") or []
            if isinstance(summary, list) and summary:
                joined = "\n\n".join(s.get("text", "") if isinstance(s, dict) else str(s) for s in summary)
                console.print(Panel(joined, title="[dim italic]reasoning[/]", border_style="dim", padding=(0, 1)))
            else:
                console.print(f"  [dim]reasoning step (id={raw.get('id', '?')})[/dim]")
        else:
            console.print(Syntax(json.dumps(raw, indent=2, default=str), "json",
                                 theme="monokai", word_wrap=True))
    elif etype == "raw_response_event":
        dtype = data.get("type", "")
        if dtype == "response.output_text.done" and data.get("text"):
            console.print(Markdown(data["text"]))
        elif dtype == "response.shell_call_command.added" and data.get("command"):
            console.print(Syntax(data["command"], "bash", theme="monokai", word_wrap=True))
        else:
            console.print(Syntax(json.dumps(data, indent=2, default=str), "json",
                                 theme="monokai", word_wrap=True))
    elif etype == "agent_updated_stream_event":
        console.print(f"  new agent: [bold]{msg.get('new_agent', '?')}[/]")
    elif etype == "ResultMessage":
        result = str(msg.get("result", ""))
        console.print(Panel(Markdown(result) if result else Text("(empty)", style="dim"),
                            title="Result", border_style="magenta", padding=(1, 2)))
    elif etype == "StatusMessage":
        status = str(msg.get("status", ""))
        console.print(Panel(status or Text("(empty)", style="dim"),
                            title="Status", border_style="magenta", padding=(1, 2)))

    console.print()


def _show_openai_events(events: list[dict]) -> None:
    """Interactive event browser for the OpenAI Agents SDK trace."""
    TYPE_STYLES = {
        "run_item_stream_event": "green",
        "raw_response_event": "dim",
        "agent_updated_stream_event": "yellow",
        "ResultMessage": "bold magenta",
    }

    while True:
        table = Table(title="OpenAI Trace", show_lines=False, expand=True)
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Type", width=14)
        table.add_column("Kind", max_width=40, style="bold", no_wrap=True, overflow="ellipsis")
        table.add_column("Preview", ratio=1, overflow="ellipsis", no_wrap=True)

        for i, msg in enumerate(events):
            etype = msg.get("type", "?")
            short, kind, preview = _openai_event_preview(msg)
            style = TYPE_STYLES.get(etype, "")
            table.add_row(str(i), Text(short, style=style), kind, preview)

        console.print(table)
        choice = Prompt.ask(
            f"\n[bold]# or range (e.g. 3-7) to inspect (0-{len(events) - 1}), or 'b' to go back[/bold]",
            default="b"
        )
        indices = _parse_msg_selection(choice, len(events))
        if indices is None:
            return
        if not indices:
            continue

        queue = list(indices)
        while queue:
            idx = queue.pop(0)
            msg = events[idx]
            queued_hint = f"  [dim]({len(queue)} more queued)[/dim]" if queue else ""
            _render_openai_event(msg, idx)
            if queued_hint:
                console.print(queued_hint)
            prompt_parts = ["[bold]"]
            prompt_parts.append("Enter=next, " if queue else "Enter=list, ")
            prompt_parts.append("#/range, 'j'=json, 's'=skip, 'b'=back[/bold]")
            nxt = Prompt.ask("".join(prompt_parts), default="").strip()
            if nxt.lower() in ("b", "back", "q"):
                return
            if nxt.lower() == "s":
                break
            if nxt.lower() == "j":
                console.print(Panel(
                    Syntax(json.dumps(msg, indent=2, default=str), "json", theme="monokai", word_wrap=True),
                    title=f"Event #{idx} (raw JSON)", border_style="blue",
                ))
                queue.insert(0, idx)
                Prompt.ask("[dim]Press Enter to continue[/dim]", default="")
                queue.pop(0)
                continue
            if nxt == "":
                continue
            new_indices = _parse_msg_selection(nxt, len(events))
            if new_indices is None:
                return
            if new_indices:
                queue = list(new_indices)


def _oh_content_text(content) -> str:
    """Flatten an OpenHands message/observation content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = content.get("content", "")
    if isinstance(content, list):
        return "\n".join(
            str(c.get("text", "")) for c in content
            if isinstance(c, dict) and c.get("text")
        )
    return str(content or "")


def _oh_action_arg(action: dict) -> tuple[str, str]:
    """Return (arg_key, value) most representative of an OpenHands action."""
    if not isinstance(action, dict):
        return "", str(action)
    for key in ("command", "message", "path", "file_text", "old_str", "new_str", "thought", "text"):
        if action.get(key):
            return key, str(action[key])
    # Fall back to the first non-bookkeeping field.
    for key, val in action.items():
        if key not in ("kind",) and val not in (None, "", [], {}):
            return key, str(val)
    return "", ""


def _openhands_event_preview(msg: dict) -> tuple[str, str, str]:
    """Extract (short_type, kind, preview) from an OpenHands event."""
    etype = msg.get("type", "?")
    short = etype.replace("Event", "").replace("Message", "")
    kind, preview = "", ""

    if etype == "ActionEvent":
        tool = msg.get("tool_name", "action")
        kind = f"tool:{tool}"
        arg_key, val = _oh_action_arg(msg.get("action") or {})
        preview = f"{arg_key}={val}" if arg_key else val
    elif etype == "ObservationEvent":
        kind = f"obs:{msg.get('tool_name', '?')}"
        preview = _oh_content_text(msg.get("observation"))
    elif etype == "MessageEvent":
        lm = msg.get("llm_message") or {}
        kind = lm.get("role", msg.get("source", "message"))
        preview = _oh_content_text(lm.get("content"))
    elif etype == "SystemPromptEvent":
        kind = "system"
        preview = str(msg.get("system_prompt", ""))
    elif etype == "ResultMessage":
        kind = "result"
        cost = msg.get("total_cost_usd", msg.get("cost_usd"))  # cost_usd: legacy traces
        preview = (f"${cost:.4f}  " if isinstance(cost, (int, float)) else "") + str(msg.get("result", ""))
    elif etype == "StatusMessage":
        kind = "status"
        preview = str(msg.get("status", ""))
    else:
        preview = str(msg.get("source", ""))

    preview = preview.replace("\n", " ").strip()
    if len(preview) > 90:
        preview = preview[:90] + "…"
    return short, kind, preview


def _render_openhands_event(msg: dict, idx: int) -> None:
    """Render a single OpenHands event in a human-friendly format."""
    etype = msg.get("type", "?")
    type_label = {
        "ActionEvent": "[yellow bold]Action[/]",
        "ObservationEvent": "[cyan]Observation[/]",
        "MessageEvent": "[green bold]Message[/]",
        "SystemPromptEvent": "[dim]SystemPrompt[/]",
        "ResultMessage": "[magenta bold]Result[/]",
        "StatusMessage": "[magenta]Status[/]",
    }.get(etype, etype)
    source = msg.get("source", "")
    header = f"#{idx}  {type_label}"
    if source:
        header += f"  [dim]({source})[/dim]"

    console.print()
    console.rule(header)

    if etype == "ActionEvent":
        reasoning = msg.get("reasoning_content")
        if reasoning:
            console.print(Panel(str(reasoning), title="[dim italic]reasoning[/]", border_style="dim", padding=(0, 1)))
        console.print(f"  [bold yellow]{msg.get('tool_name', 'action')}[/]")
        action = msg.get("action") or {}
        if isinstance(action, dict):
            for k, v in action.items():
                if k == "kind" or v in (None, "", [], {}):
                    continue
                v_str = str(v)
                lang = "bash" if k == "command" else ("lean4" if ("import" in v_str or "theorem" in v_str) else "text")
                if "\n" in v_str or len(v_str) > 100:
                    console.print(f"    [dim]{k}:[/]")
                    console.print(Syntax(v_str[:4000], lang, theme="monokai", word_wrap=True, padding=(0, 1)))
                else:
                    console.print(f"    [dim]{k}:[/] {v_str}")
    elif etype == "ObservationEvent":
        out = _oh_content_text(msg.get("observation"))
        if len(out) > 4000:
            console.print(f"  [dim]({len(out)} chars — showing first 4000)[/dim]")
            out = out[:4000] + "…"
        console.print(Syntax(out or "(empty)", "text", theme="monokai", word_wrap=True, padding=(0, 1)))
    elif etype == "MessageEvent":
        lm = msg.get("llm_message") or {}
        text = _oh_content_text(lm.get("content"))
        console.print(Markdown(text) if text else Text("(empty)", style="dim"))
    elif etype == "SystemPromptEvent":
        sp = str(msg.get("system_prompt", ""))
        console.print(Panel(sp, title="[dim]system prompt[/]", border_style="dim", padding=(0, 1)))
    elif etype == "ResultMessage":
        result = str(msg.get("result", ""))
        usage = msg.get("usage") or {}
        cost = msg.get("total_cost_usd", msg.get("cost_usd"))  # cost_usd: legacy traces
        meta = []
        if isinstance(cost, (int, float)):
            meta.append(f"cost ${cost:.4f}")
        tu = usage.get("token_usage") if isinstance(usage, dict) else None
        if isinstance(tu, dict):
            meta.append(f"prompt {tu.get('prompt_tokens', '?')} / completion {tu.get('completion_tokens', '?')}")
        if meta:
            console.print(f"  [dim]{'   '.join(meta)}[/dim]")
        console.print(Panel(Markdown(result) if result else Text("(empty)", style="dim"),
                            title="Result", border_style="magenta", padding=(1, 2)))
    elif etype == "StatusMessage":
        status = str(msg.get("status", ""))
        console.print(Panel(status or Text("(empty)", style="dim"),
                            title="Status", border_style="magenta", padding=(1, 2)))
    else:
        console.print(Syntax(json.dumps(msg, indent=2, default=str), "json", theme="monokai", word_wrap=True))

    console.print()


def _codex_event_preview(msg: dict) -> tuple[str, str, str]:
    """Extract (short_type, kind, preview) from a Codex notification."""
    etype = msg.get("type", "?")
    if etype in ("ResultMessage", "StatusMessage"):
        kind = "result" if etype == "ResultMessage" else "status"
        preview = str(msg.get("result", msg.get("status", "")))
    else:
        method = msg.get("method", "")
        payload = msg.get("payload") or {}
        ptype = payload.get("type", "")
        short = (method or ptype).split(".")[-1]
        kind, preview = short, ""
        item = payload.get("item") or {}
        if isinstance(item, dict) and item:
            itype = item.get("type", "")
            if itype == "agentMessage":
                kind, preview = "text", str(item.get("text", ""))
            elif item.get("command"):
                kind, preview = "command", str(item.get("command"))
            else:
                kind, preview = itype or short, item.get("id", "")
        tu = payload.get("token_usage")
        if isinstance(tu, dict):
            kind = "tokens"
            preview = f"in:{tu.get('input_tokens', '?')} out:{tu.get('output_tokens', '?')}"
        etype = short  # display the notification variant, not the wrapper name

    preview = str(preview).replace("\n", " ").strip()
    if len(preview) > 90:
        preview = preview[:90] + "…"
    return etype, kind, preview


def _render_codex_event(msg: dict, idx: int) -> None:
    """Render a single Codex notification in a human-friendly format."""
    etype = msg.get("type", "?")
    if etype == "ResultMessage":
        result = str(msg.get("result", ""))
        console.print()
        console.rule(f"#{idx}  [magenta bold]Result[/]")
        console.print(Panel(Markdown(result) if result else Text("(empty)", style="dim"),
                            title="Result", border_style="magenta", padding=(1, 2)))
        console.print()
        return
    if etype == "StatusMessage":
        status = str(msg.get("status", ""))
        console.print()
        console.rule(f"#{idx}  [magenta]Status[/]")
        console.print(Panel(status or Text("(empty)", style="dim"),
                            title="Status", border_style="magenta", padding=(1, 2)))
        console.print()
        return

    method = msg.get("method", "")
    payload = msg.get("payload") or {}
    short = (method or payload.get("type", "?")).split(".")[-1]
    console.print()
    console.rule(f"#{idx}  [green bold]{short}[/]")
    item = payload.get("item") or {}
    if isinstance(item, dict) and item.get("type") == "agentMessage":
        console.print(Markdown(str(item.get("text", "")) or "(empty)"))
    elif isinstance(item, dict) and item.get("command"):
        console.print(Syntax(str(item.get("command")), "bash", theme="monokai", word_wrap=True, padding=(0, 1)))
    else:
        console.print(Syntax(json.dumps(payload, indent=2, default=str), "json", theme="monokai", word_wrap=True))
    console.print()


def _show_event_browser(events: list[dict], title: str, preview_fn, render_fn,
                        type_styles: dict) -> None:
    """Generic interactive event browser (shared by the openhands / codex traces)."""
    while True:
        table = Table(title=title, show_lines=False, expand=True)
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Type", width=16)
        table.add_column("Kind", max_width=40, style="bold", no_wrap=True, overflow="ellipsis")
        table.add_column("Preview", ratio=1, overflow="ellipsis", no_wrap=True)

        for i, msg in enumerate(events):
            short, kind, preview = preview_fn(msg)
            style = type_styles.get(msg.get("type", ""), "")
            table.add_row(str(i), Text(short, style=style), kind, preview)

        console.print(table)
        choice = Prompt.ask(
            f"\n[bold]# or range (e.g. 3-7) to inspect (0-{len(events) - 1}), or 'b' to go back[/bold]",
            default="b"
        )
        indices = _parse_msg_selection(choice, len(events))
        if indices is None:
            return
        if not indices:
            continue

        queue = list(indices)
        while queue:
            idx = queue.pop(0)
            msg = events[idx]
            queued_hint = f"  [dim]({len(queue)} more queued)[/dim]" if queue else ""
            render_fn(msg, idx)
            if queued_hint:
                console.print(queued_hint)
            prompt_parts = ["[bold]", "Enter=next, " if queue else "Enter=list, ",
                            "#/range, 'j'=json, 's'=skip, 'b'=back[/bold]"]
            nxt = Prompt.ask("".join(prompt_parts), default="").strip()
            if nxt.lower() in ("b", "back", "q"):
                return
            if nxt.lower() == "s":
                break
            if nxt.lower() == "j":
                console.print(Panel(
                    Syntax(json.dumps(msg, indent=2, default=str), "json", theme="monokai", word_wrap=True),
                    title=f"Event #{idx} (raw JSON)", border_style="blue",
                ))
                queue.insert(0, idx)
                Prompt.ask("[dim]Press Enter to continue[/dim]", default="")
                queue.pop(0)
                continue
            if nxt == "":
                continue
            new_indices = _parse_msg_selection(nxt, len(events))
            if new_indices is None:
                return
            if new_indices:
                queue = list(new_indices)


def _show_openhands_events(events: list[dict]) -> None:
    """Interactive event browser for the OpenHands SDK trace."""
    _show_event_browser(events, "OpenHands Trace", _openhands_event_preview, _render_openhands_event, {
        "ActionEvent": "yellow", "ObservationEvent": "cyan", "MessageEvent": "green",
        "SystemPromptEvent": "dim", "ResultMessage": "bold magenta", "StatusMessage": "magenta",
    })


def _show_codex_events(events: list[dict]) -> None:
    """Interactive event browser for the Codex SDK trace."""
    _show_event_browser(events, "Codex Trace", _codex_event_preview, _render_codex_event, {
        "ResultMessage": "bold magenta", "StatusMessage": "magenta",
    })


def _show_claude_messages(messages: list[dict]) -> None:
    """Interactive message browser for the Claude conversation."""
    TYPE_STYLES = {
        "AssistantMessage": "green",
        "UserMessage": "cyan",
        "SystemMessage": "yellow",
        "RateLimitEvent": "dim",
        "ResultMessage": "bold magenta",
    }

    while True:
        table = Table(title="Claude Conversation", show_lines=False, expand=True)
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Type", width=12)
        table.add_column("Kind", max_width=40, style="bold", no_wrap=True, overflow="ellipsis")
        table.add_column("Preview", ratio=1, overflow="ellipsis", no_wrap=True)

        for i, msg in enumerate(messages):
            mtype = msg.get("type", "?")
            subtype = msg.get("subtype", "")
            short_type = {
                "AssistantMessage": "Assistant",
                "UserMessage": "User",
                "SystemMessage": "System",
                "RateLimitEvent": "RateLimit",
                "ResultMessage": "Result",
            }.get(mtype, mtype)
            if subtype:
                short_type += f"/{subtype}"

            style = TYPE_STYLES.get(mtype, "")
            kind, tokens, preview = _msg_preview(msg)

            table.add_row(
                str(i),
                Text(short_type, style=style),
                kind,
                preview,
            )

        console.print(table)
        choice = Prompt.ask(
            f"\n[bold]# or range (e.g. 3-7) to inspect (0-{len(messages) - 1}), or 'b' to go back[/bold]",
            default="b"
        )
        indices = _parse_msg_selection(choice, len(messages))
        if indices is None:
            return
        if not indices:
            continue

        # Walk through selected messages, allowing jumps
        queue = list(indices)
        while queue:
            idx = queue.pop(0)
            msg = messages[idx]
            queued_hint = f"  [dim]({len(queue)} more queued)[/dim]" if queue else ""
            _render_message(msg, idx)
            if queued_hint:
                console.print(queued_hint)

            prompt_parts = ["[bold]"]
            if queue:
                prompt_parts.append("Enter=next, ")
            else:
                prompt_parts.append("Enter=list, ")
            prompt_parts.append(f"#/range, 'j'=json, 's'=skip, 'b'=back[/bold]")
            nxt = Prompt.ask("".join(prompt_parts), default="")
            nxt = nxt.strip()
            if nxt.lower() in ("b", "back", "q"):
                return
            if nxt.lower() == "s":
                break  # skip remaining queued, back to table
            if nxt.lower() == "j":
                # Show raw JSON for current message, then re-prompt
                console.print(Panel(
                    Syntax(json.dumps(msg, indent=2, default=str), "json", theme="monokai", word_wrap=True),
                    title=f"Message #{idx} (raw JSON)",
                    border_style="blue",
                ))
                queue.insert(0, idx)  # re-queue current so next Enter moves forward
                Prompt.ask("[dim]Press Enter to continue[/dim]", default="")
                queue.pop(0)  # consume the re-queued entry
                continue
            if nxt == "":
                continue  # next in queue, or back to table if empty
            # Parse new selection and replace the queue
            new_indices = _parse_msg_selection(nxt, len(messages))
            if new_indices is None:
                return
            if new_indices:
                queue = list(new_indices)


MENU_LABELS = {1: "Text", 2: "Solution", 3: "Prompt", 4: "Result"}


def _get_content(problem: Problem, choice: int) -> str | list[str] | None:
    """Extract the raw content for a given menu choice from a problem."""
    if choice == 1:
        return problem.text
    elif choice == 2:
        return problem.solution
    elif choice == 3:
        return str(problem.metadata.get("prompt", "(none)"))
    elif choice == 4:
        return str(problem.metadata.get("result", "(none)"))
    return None


def _make_entry_panel(content: str | None, title: str, choice: int, cw: int | None) -> Panel:
    """Build a Panel for a single string entry (one element of a list, or the whole content)."""
    border = "blue" if choice in (1, 2) else "green"
    if choice == 4:
        inner = Markdown(content or "")
        padding = (1, 2)
    elif choice in (1, 2):
        inner = Syntax(content or "", "lean4", theme="monokai", line_numbers=True, word_wrap=True)
        padding = (0, 1)
    else:
        inner = content or ""
        padding = (0, 1)
    kwargs: dict = dict(title=f"[bold]{title}[/bold]", border_style=border, padding=padding)
    if cw:
        kwargs["width"] = cw
    return Panel(inner, **kwargs)


def _show_items(problems: list[Problem], choices: list[int]) -> None:
    """Non-interactive: print the requested menu items in order, then exit."""
    n = len(problems)

    # Print headers
    if n == 1:
        console.print(Panel(_header(problems[0]), title="Problem", border_style="bright_blue"))
    else:
        cw = console.width // n
        headers = [Panel(_header(p), title=f"Problem — {p.info.task}", border_style="bright_blue", width=cw)
                   for p in problems]
        _print_side_by_side(headers, cw)

    for choice in choices:
        label = MENU_LABELS.get(choice, str(choice))
        # console.rule(f"[bold]{label.upper()}[/bold]")

        contents = [_get_content(p, choice) for p in problems]

        # Determine if any content is a list (multi-entry)
        is_list = [isinstance(c, list) for c in contents]

        if n == 1:
            c = contents[0]
            if isinstance(c, list):
                for i, entry in enumerate(c):
                    # console.rule(f"[dim]entry {i + 1} / {len(c)}[/dim]", characters="─")
                    console.print(_make_entry_panel(entry, problems[0].info.task, choice, None))
            else:
                console.print(_make_entry_panel(c, problems[0].info.task, choice, None))
        else:
            cw = console.width // n
            if not any(is_list):
                # All scalar — simple side-by-side row
                panels = [_make_entry_panel(c, p.info.task, choice, cw)  # type: ignore[arg-type]
                          for c, p in zip(contents, problems)]
                _print_side_by_side(panels, cw)
            else:
                # At least one is a list — iterate over max entries, pad missing with None
                max_len = max(len(c) if isinstance(c, list) else 1 for c in contents)
                for i in range(max_len):
                    # console.rule(f"[dim]entry {i + 1} / {max_len}[/dim]", characters="─")
                    panels = []
                    for c, p in zip(contents, problems):
                        if isinstance(c, list):
                            entry = c[i] if i < len(c) else None
                        else:
                            entry = c if i == 0 else None  # type: ignore[assignment]
                        panels.append(_make_entry_panel(entry, p.info.task, choice, cw))
                    _print_side_by_side(panels, cw)


def _render_column_text(content: str | list[str] | None, title: str, col_width: int, lang: str = "lean4") -> Panel:
    """Render a text field as a Rich Panel suitable for column layout."""
    if not content:
        inner = Text("(empty)", style="dim")
    elif isinstance(content, list):
        inner = Group(*[
            Syntax(entry or "", lang, theme="monokai", line_numbers=True, word_wrap=True)
            for entry in content
        ])
    else:
        inner = Syntax(content, lang, theme="monokai", line_numbers=True, word_wrap=True)
    return Panel(inner, title=f"[bold]{title}[/bold]", border_style="blue", width=col_width)


def _print_side_by_side(renderables: list, col_width: int) -> None:
    """Print a list of renderables as equal-width columns using a Table."""
    tbl = Table.grid(padding=0, expand=False)
    for _ in renderables:
        tbl.add_column(width=col_width, no_wrap=False)
    tbl.add_row(*renderables)
    console.print(tbl)


def explore_multi(problems: list[Problem]) -> None:
    """Interactive TUI for exploring multiple Problem objects side by side."""
    labels = [p.info.task for p in problems]
    n = len(problems)

    def col_width() -> int:
        return console.width // n

    while True:
        cw = col_width()
        headers = [Panel(_header(p), title=f"Problem — {p.info.task}", border_style="bright_blue", width=cw)
                   for p in problems]
        _print_side_by_side(headers, cw)

        items = {
            "1": "Text",
            "2": "Solution",
            "3": "Prompt",
            "4": "Result",
            "q": "Quit",
        }
        for key, label in items.items():
            console.print(f"  [bold cyan]{key}[/bold cyan]  {label}")

        choice = Prompt.ask("\n[bold]Select[/bold]", default="q")

        if choice in ("q", "quit"):
            break
        elif choice == "1":
            cw = col_width()
            cols = [_render_column_text(p.text, lbl, cw) for p, lbl in zip(problems, labels)]
            _print_side_by_side(cols, cw)
        elif choice == "2":
            cw = col_width()
            cols = [_render_column_text(p.solution, lbl, cw) for p, lbl in zip(problems, labels)]
            _print_side_by_side(cols, cw)
        elif choice == "3":
            cw = col_width()
            cols = [Panel(str(p.metadata.get("prompt", "(none)")), title=lbl, border_style="green", width=cw)
                    for p, lbl in zip(problems, labels)]
            _print_side_by_side(cols, cw)
        elif choice == "4":
            cw = col_width()
            cols = [Panel(Markdown(str(p.metadata.get("result", "(none)"))), title=lbl, border_style="green", width=cw,
                          padding=(1, 2))
                    for p, lbl in zip(problems, labels)]
            _print_side_by_side(cols, cw)
        else:
            console.print("[red]Invalid selection[/red]")
            continue

        Prompt.ask("[dim]Press Enter to continue[/dim]", default="")


# Trace metadata keys, one per -p backend, grouped by trace *format*. Each SDK
# streams a different event shape, so each family gets its own renderer:
#   * claude    — Claude Agent SDK messages
#   * openai    — OpenAI Agents SDK stream events (base OpenAI + opencode)
#   * codex     — OpenAI Codex SDK notifications
#   * openhands — OpenHands SDK events (base + the named model variants)
CLAUDE_PROVERS = ("claude",)
OPENAI_PROVERS = ("openai", "opencode")
CODEX_PROVERS = ("codex",)
OPENHANDS_PROVERS = ("openhands", "kimi", "minimax", "glm")
PROVER_TRACE_KEYS = CLAUDE_PROVERS + OPENAI_PROVERS + CODEX_PROVERS + OPENHANDS_PROVERS


def _detect_prover(problem: Problem) -> str | None:
    """Return the prover key whose trace is present in metadata, if any."""
    for key in PROVER_TRACE_KEYS:
        if isinstance(problem.metadata.get(key), list) and problem.metadata[key]:
            return key
    return None


def explore(problem: Problem) -> None:
    """Interactive TUI menu for exploring a Problem object."""
    meta = problem.metadata
    prover_name = _detect_prover(problem)
    trace = meta.get(prover_name, []) if prover_name else []
    n_trace = len(trace) if isinstance(trace, list) else 0

    # Build menu items dynamically
    known_meta = {"prompt", "status", "result", *PROVER_TRACE_KEYS}
    extra_keys = [k for k in meta if k not in known_meta]

    while True:
        console.print(Panel(_header(problem), title="Problem", border_style="bright_blue"))

        text_label = "Text"
        if isinstance(problem.text, list):
            text_label += f" ({len(problem.text)} entries)"
        sol_label = "Solution"
        if isinstance(problem.solution, list):
            sol_label += f" ({len(problem.solution)} entries)"
        trace_label = None
        if n_trace:
            label_name = "Claude conversation" if prover_name in CLAUDE_PROVERS else f"{prover_name} trace"
            trace_label = f"{label_name} ({n_trace} entries)"
        items = {
            "1": text_label,
            "2": sol_label,
            "3": "Prompt",
            "4": "Result",
            "5": trace_label,
        }
        # Add extra metadata keys
        extra_start = 6
        for i, k in enumerate(extra_keys):
            items[str(extra_start + i)] = f"Metadata: {k}"

        for key, label in items.items():
            if label:
                console.print(f"  [bold cyan]{key}[/bold cyan]  {label}")
        console.print(f"  [bold cyan]q[/bold cyan]  Quit")

        choice = Prompt.ask("\n[bold]Select[/bold]", default="q")

        if choice in ("q", "quit"):
            break
        elif choice == "1":
            _show_text(problem.text, "Text")
        elif choice == "2":
            _show_text(problem.solution, "Solution")
        elif choice == "3":
            console.print(Panel.fit("Prompt", border_style="green"))
            prompt = str(meta.get("prompt", "(none)"))
            console.print(f"  {prompt}")
        elif choice == "4":
            result = str(meta.get("result", "(none)"))
            console.print(Panel(Markdown(result), title="Result", border_style="green", padding=(1, 2)))
        elif choice == "5" and n_trace:
            if prover_name in CLAUDE_PROVERS:
                _show_claude_messages(trace)
            elif prover_name in OPENHANDS_PROVERS:
                _show_openhands_events(trace)
            elif prover_name in CODEX_PROVERS:
                _show_codex_events(trace)
            else:
                _show_openai_events(trace)
            continue  # message browser has its own pause
        else:
            # Check extra metadata keys
            idx = int(choice) - extra_start if choice.isdigit() else -1
            if 0 <= idx < len(extra_keys):
                k = extra_keys[idx]
                val = meta[k]
                console.print(Panel(
                    Syntax(json.dumps(val, indent=2, default=str), "json", theme="monokai", word_wrap=True),
                    title=f"Metadata: {k}", border_style="green",
                ))
            else:
                console.print("[red]Invalid selection[/red]")
                continue

        Prompt.ask("[dim]Press Enter to continue[/dim]", default="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--prover', default='claude', type=str, choices=sorted(PROVER_TRACE_KEYS),
                        help='Prover backend whose results to display (default: claude)')
    parser.add_argument('-t', '--task', default=None, nargs='+', choices=sorted(TASKS.keys()),
                        help='Task name(s) to display. Specify multiple to compare side by side.')
    parser.add_argument('-i', '--ids', type=int, nargs='+', default=None, help='Problem ID to display')
    parser.add_argument("--attempt", type=int, default=0, help="Attempt number (default: 0)")
    parser.add_argument("--base-dir", type=Path, default=None,
                        help="Base directory for problem data (default: data/<prover>)")
    parser.add_argument("--show", type=int, nargs='+', metavar='N',
                        help='Menu item numbers to print non-interactively and exit (e.g. --show 4 2)')

    args = parser.parse_args()
    if args.base_dir is None:
        args.base_dir = var.DATA_DIRECTORY / args.prover
    pids = iter(args.ids if args.ids is not None else [int(input("  ID >  ")), ])
    pid = None
    if args.task:
        tasks = args.task
    else:
        tasks = [input("TASK >  ").upper().replace('-', '_')]
    while True:
        if pid is not None:
            input(f"Press Enter for next problem: ")
            # Clear screen including scrollback buffer (equivalent to the shell `clear` command)
            print("\033[3J\033[H\033[2J", end="")
        try:
            pid = next(pids)
        except StopIteration:
            break
        problems = [
            Problem.load(info=ProblemInfo(id=pid, task=task, **TASKS[task]['dataset'], attempt=args.attempt),
                         base_dir=args.base_dir)
            for task in tasks
        ]
        if args.show:
            _show_items(problems, args.show)
        elif len(problems) == 1:
            explore(problems[0])
        else:
            explore_multi(problems)
