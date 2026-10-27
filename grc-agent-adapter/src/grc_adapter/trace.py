"""A call's conversation as one row of Qoherent's SFT row format, and as readable text.

The row has the release's keys in the release's order, as grc-agent-headless writes its
trace.jsonl (GRC-Bench tier_a/trace.py), so sessions from macOS and from Linux read the same:
    session_id, task_id, category, verdict, teacher, system, tools, messages, usage
followed by this adapter's metadata under "grc_adapter".

Message rules (same as grc-agent-headless): user prompts when text; tool returns and retry
prompts are "tool" messages (role, tool_call_id, name, content); assistant messages are (role,
content, tool_calls, thinking) with content "" when there are only tool calls, arguments as the
string the model emitted, thinking in a parallel field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _args_text(args: Any) -> str:
    if isinstance(args, str):
        return args
    return "{}" if args is None else json.dumps(args, ensure_ascii=False)


def _content_text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, default=str)


def map_messages(messages: list[Any], *, include_thinking: bool = True) -> list[dict[str, Any]]:
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        RetryPromptPart,
        TextPart,
        ThinkingPart,
        ToolCallPart,
        ToolReturnPart,
    )
    out: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if part.__class__.__name__ == "UserPromptPart" and isinstance(getattr(part, "content", None), str):
                    out.append({"role": "user", "content": part.content})
                elif isinstance(part, ToolReturnPart):
                    out.append({"role": "tool", "tool_call_id": part.tool_call_id,
                                "name": part.tool_name, "content": _content_text(part.content)})
                elif isinstance(part, RetryPromptPart):
                    out.append({"role": "tool", "tool_call_id": part.tool_call_id,
                                "name": part.tool_name or "unknown", "content": _content_text(part.content)})
        elif isinstance(msg, ModelResponse):
            thinking = "".join(p.content for p in msg.parts if isinstance(p, ThinkingPart)).strip()
            text = "".join(p.content for p in msg.parts if isinstance(p, TextPart)).strip()
            calls = [{"id": c.tool_call_id, "name": c.tool_name, "arguments": _args_text(c.args)}
                     for c in msg.parts if isinstance(c, ToolCallPart)]
            if not text and not calls and not (include_thinking and thinking):
                continue
            row: dict[str, Any] = {"role": "assistant", "content": text}
            if calls:
                row["tool_calls"] = calls
            if include_thinking and thinking:
                row["thinking"] = thinking
            out.append(row)
    return out


def sum_usage(messages: list[Any]) -> dict[str, int]:
    from pydantic_ai.messages import ModelResponse
    totals: dict[str, int] = {}
    for msg in messages:
        if isinstance(msg, ModelResponse) and msg.usage is not None:
            u = msg.usage
            for name in ("input_tokens", "output_tokens", "requests"):
                value = getattr(u, name, None)
                if isinstance(value, int):
                    totals[name] = totals.get(name, 0) + value
            details = getattr(u, "details", None)
            reasoning = (details.get("reasoning_tokens") if isinstance(details, dict)
                         else getattr(details, "reasoning_tokens", None))
            if isinstance(reasoning, int):
                totals["reasoning_tokens"] = totals.get("reasoning_tokens", 0) + reasoning
    return totals


def system_prompt(messages: list[Any]) -> str:
    from pydantic_ai.messages import ModelRequest
    return next((m.instructions for m in messages
                 if isinstance(m, ModelRequest) and getattr(m, "instructions", None)), None) or ""


def verdict(status: str) -> str:
    return {"infrastructure_error": "infrastructure_abort", "cap_failure": "cap_failure",
            "agent_error": "agent_error"}.get(status, "ungraded")


def session_row(*, session: str, call: int, status: str, model: dict[str, Any], messages: list[Any],
                tools: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "session_id": f"grc-adapter-{session}-call-{call:02d}",
        "task_id": f"adhoc-{session}",
        "category": "adhoc",
        "verdict": verdict(status),
        "teacher": {"provider": model.get("provider"), "model": model.get("model_name")},
        "system": system_prompt(messages),
        "tools": [{"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
                  for t in tools],
        "messages": map_messages(messages),
        "usage": sum_usage(messages),
    }
    row["grc_adapter"] = meta
    return row


def write_trace(path: Path, row: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row) + "\n")


def transcript(rows: list[dict[str, Any]], title: str) -> str:
    """The mapped messages as plain markdown text."""
    out = [f"# {title}", ""]
    for m in rows:
        role = m["role"]
        if role == "user":
            out += ["## user", "", m["content"], ""]
        elif role == "assistant":
            out.append("## assistant")
            out.append("")
            if m.get("thinking"):
                out += ["<details><summary>thinking</summary>", "", m["thinking"], "", "</details>", ""]
            if m["content"]:
                out += [m["content"], ""]
            for c in m.get("tool_calls") or []:
                out += [f"**tool call** `{c['name']}` ({c['id']})", "", "```json", c["arguments"], "```", ""]
        else:
            out += [f"**tool result** `{m['name']}` ({m['tool_call_id']})", "", "```", m["content"], "```", ""]
    return "\n".join(out)
